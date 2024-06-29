# `D:\src\scipysrc\pandas\pandas\tests\strings\test_split_partition.py`

```
# 从 datetime 模块导入 datetime 类
from datetime import datetime
# 导入 re 模块，用于正则表达式操作
import re

# 导入 numpy 库并将其命名为 np
import numpy as np
# 导入 pytest 库
import pytest

# 导入 pandas 库并将其命名为 pd
import pandas as pd
# 从 pandas 中导入 DataFrame、Index、MultiIndex、Series 和 _testing 模块
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
)
# 从 pandas.tests.strings 模块中导入 _convert_na_value 和 object_pyarrow_numpy
from pandas.tests.strings import (
    _convert_na_value,
    object_pyarrow_numpy,
)


# 使用 pytest.mark.parametrize 装饰器，参数化 method 参数为 "split" 和 "rsplit"
@pytest.mark.parametrize("method", ["split", "rsplit"])
# 定义测试函数 test_split，参数为 any_string_dtype 和 method
def test_split(any_string_dtype, method):
    # 创建 Series 对象 values，包含字符串数组和 NaN 值，使用指定的 dtype
    values = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)

    # 调用 getattr 函数，获取 values.str 中的 method 方法（split 或 rsplit），使用 "_" 作为分隔符
    result = getattr(values.str, method)("_")
    # 创建期望的结果 Series exp，包含分割后的字符串列表，将 NaN 值转换为指定的缺失值
    exp = Series([["a", "b", "c"], ["c", "d", "e"], np.nan, ["f", "g", "h"]])
    exp = _convert_na_value(values, exp)
    # 使用 tm.assert_series_equal 函数比较 result 和 exp，确保它们相等
    tm.assert_series_equal(result, exp)


# 使用 pytest.mark.parametrize 装饰器，参数化 method 参数为 "split" 和 "rsplit"
@pytest.mark.parametrize("method", ["split", "rsplit"])
# 定义测试函数 test_split_more_than_one_char，参数为 any_string_dtype 和 method
def test_split_more_than_one_char(any_string_dtype, method):
    # 创建 Series 对象 values，包含更长的字符串和 NaN 值，使用指定的 dtype
    values = Series(["a__b__c", "c__d__e", np.nan, "f__g__h"], dtype=any_string_dtype)
    
    # 调用 getattr 函数，获取 values.str 中的 method 方法（split 或 rsplit），使用 "__" 作为分隔符
    result = getattr(values.str, method)("__")
    # 创建期望的结果 Series exp，包含分割后的字符串列表，将 NaN 值转换为指定的缺失值
    exp = Series([["a", "b", "c"], ["c", "d", "e"], np.nan, ["f", "g", "h"]])
    exp = _convert_na_value(values, exp)
    # 使用 tm.assert_series_equal 函数比较 result 和 exp，确保它们相等
    tm.assert_series_equal(result, exp)

    # 再次调用 getattr 函数，获取 values.str 中的 method 方法（split 或 rsplit），使用 "__" 作为分隔符，并设置 expand=False
    result = getattr(values.str, method)("__", expand=False)
    # 使用 tm.assert_series_equal 函数比较 result 和 exp，确保它们相等
    tm.assert_series_equal(result, exp)


# 定义测试函数 test_split_more_regex_split，参数为 any_string_dtype
def test_split_more_regex_split(any_string_dtype):
    # 创建 Series 对象 values，包含正则表达式分割后的字符串和 NaN 值，使用指定的 dtype
    values = Series(["a,b_c", "c_d,e", np.nan, "f,g,h"], dtype=any_string_dtype)
    # 调用 values.str.split 函数，使用正则表达式 "[,_]" 进行分割
    result = values.str.split("[,_]")
    # 创建期望的结果 Series exp，包含分割后的字符串列表，将 NaN 值转换为指定的缺失值
    exp = Series([["a", "b", "c"], ["c", "d", "e"], np.nan, ["f", "g", "h"]])
    exp = _convert_na_value(values, exp)
    # 使用 tm.assert_series_equal 函数比较 result 和 exp，确保它们相等
    tm.assert_series_equal(result, exp)


# 定义测试函数 test_split_regex，参数为 any_string_dtype
def test_split_regex(any_string_dtype):
    # GH 43563
    # 创建 Series 对象 values，包含指定的字符串和 NaN 值，使用指定的 dtype
    values = Series("xxxjpgzzz.jpg", dtype=any_string_dtype)
    # 调用 values.str.split 函数，使用正则表达式 r"\.jpg" 进行分割，regex=True 表示显式使用正则表达式
    result = values.str.split(r"\.jpg", regex=True)
    # 创建期望的结果 Series exp，包含分割后的字符串列表
    exp = Series([["xxxjpgzzz", ""]])
    # 使用 tm.assert_series_equal 函数比较 result 和 exp，确保它们相等
    tm.assert_series_equal(result, exp)


# 定义测试函数 test_split_regex_explicit，参数为 any_string_dtype
def test_split_regex_explicit(any_string_dtype):
    # explicit regex = True split with compiled regex
    # 编译正则表达式模式
    regex_pat = re.compile(r".jpg")
    # 创建 Series 对象 values，包含指定的字符串和 NaN 值，使用指定的 dtype
    values = Series("xxxjpgzzz.jpg", dtype=any_string_dtype)
    # 调用 values.str.split 函数，使用编译后的正则表达式 regex_pat 进行分割
    result = values.str.split(regex_pat)
    # 创建期望的结果 Series exp，包含分割后的字符串列表
    exp = Series([["xx", "zzz", ""]])
    # 使用 tm.assert_series_equal 函数比较 result 和 exp，确保它们相等
    tm.assert_series_equal(result, exp)

    # 再次调用 values.str.split 函数，使用 r"\.jpg" 进行分割，regex=False 表示非显式使用正则表达式
    result = values.str.split(r"\.jpg", regex=False)
    # 创建期望的结果 Series exp，包含分割后的字符串列表
    exp = Series([["xxxjpgzzz.jpg"]])
    # 使用 tm.assert_series_equal 函数比较 result 和 exp，确保它们相等
    tm.assert_series_equal(result, exp)

    # 再次调用 values.str.split 函数，使用 r"." 进行分割
    result = values.str.split(r".")
    # 创建期望的结果 Series exp，包含分割后的字符串列表
    exp = Series([["xxxjpgzzz", "jpg"]])
    # 使用 tm.assert_series_equal 函数比较 result 和 exp，确保它们相等
    tm.assert_series_equal(result, exp)

    # 再次调用 values.str.split 函数，使用 r".jpg" 进行分割
    result = values.str.split(r".jpg")
    # 创建期望的结果 Series exp，包含分割后的字符串列表
    exp = Series([["xx", "zzz", ""]])
    # 使用 tm.assert_series_equal 函数比较 result 和 exp，确保它们相等
    tm.assert_series_equal(result, exp)

    # 使用 regex=False 时，使用编译后的正则表达式 regex_pat 进行分割，预期会引发 ValueError 异常
    with pytest.raises(
        ValueError,
        match="Cannot use a compiled regex as replacement pattern with regex=False",
    ):
        values.str.split(regex_pat, regex=False)


# 使用 pytest.mark.parametrize 装饰器，参数化 expand 参数为 None 和 False
@pytest.mark.parametrize("expand", [None, False])
@pytest.mark.parametrize("method", ["split", "rsplit"])
# 使用 pytest 的 parametrize 装饰器，为 test_split_object_mixed 函数参数化两种方法：split 和 rsplit
def test_split_object_mixed(expand, method):
    # 创建一个包含多种类型对象的 Series
    mixed = Series(["a_b_c", np.nan, "d_e_f", True, datetime.today(), None, 1, 2.0])
    # 调用 mixed 对象的 str 属性中的 method 方法（split 或 rsplit），使用 "_" 作为分隔符，并根据 expand 参数进行扩展
    result = getattr(mixed.str, method)("_", expand=expand)
    # 预期的结果是一个包含各种对象类型的 Series，其中字符串根据 "_" 分割成列表，其他类型保持不变
    exp = Series(
        [
            ["a", "b", "c"],
            np.nan,
            ["d", "e", "f"],
            np.nan,
            np.nan,
            None,
            np.nan,
            np.nan,
        ]
    )
    # 断言 result 是一个 Series 对象
    assert isinstance(result, Series)
    # 使用 pytest 的 assert_almost_equal 函数比较 result 和 exp 是否近似相等
    tm.assert_almost_equal(result, exp)


@pytest.mark.parametrize("method", ["split", "rsplit"])
# 使用 pytest 的 parametrize 装饰器，为 test_split_n 函数参数化两种方法：split 和 rsplit，以及 n 参数的两种情况：None 和 0
@pytest.mark.parametrize("n", [None, 0])
def test_split_n(any_string_dtype, method, n):
    # 创建一个包含多种类型对象的 Series，每个元素都是字符串类型
    s = Series(["a b", pd.NA, "b c"], dtype=any_string_dtype)
    # 预期的结果是一个包含各种对象类型的 Series，其中字符串根据 " " 分割成列表，其他类型保持不变，根据 n 参数限制分割次数
    expected = Series([["a", "b"], pd.NA, ["b", "c"]])
    # 调用 s 对象的 str 属性中的 method 方法（split 或 rsplit），使用 " " 作为分隔符，并根据 n 参数限制分割次数
    result = getattr(s.str, method)(" ", n=n)
    # 将预期结果中的 NA 值转换为 s 对象相应的 NA 值
    expected = _convert_na_value(s, expected)
    # 使用 pytest 的 assert_series_equal 函数比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


def test_rsplit(any_string_dtype):
    # 注释：rsplit 不支持正则表达式分割
    # 创建一个包含多种类型对象的 Series，每个元素都是字符串类型
    values = Series(["a,b_c", "c_d,e", np.nan, "f,g,h"], dtype=any_string_dtype)
    # 调用 values 对象的 str 属性中的 rsplit 方法，使用 "[,_]" 作为分隔符
    result = values.str.rsplit("[,_]")
    # 预期的结果是一个包含各种对象类型的 Series，其中字符串根据 "[,_]" 分割成列表，其他类型保持不变
    exp = Series([["a,b_c"], ["c_d,e"], np.nan, ["f,g,h"]])
    # 将预期结果中的 NA 值转换为 values 对象相应的 NA 值
    exp = _convert_na_value(values, exp)
    # 使用 pytest 的 assert_series_equal 函数比较 result 和 exp 是否相等
    tm.assert_series_equal(result, exp)


def test_rsplit_max_number(any_string_dtype):
    # 设置最大分割次数为 1，确保从反向进行分割
    # 创建一个包含多种类型对象的 Series，每个元素都是字符串类型
    values = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)
    # 调用 values 对象的 str 属性中的 rsplit 方法，使用 "_" 作为分隔符，并限制最大分割次数为 1
    result = values.str.rsplit("_", n=1)
    # 预期的结果是一个包含各种对象类型的 Series，其中字符串根据 "_" 从反向进行一次分割成列表，其他类型保持不变
    exp = Series([["a_b", "c"], ["c_d", "e"], np.nan, ["f_g", "h"]])
    # 将预期结果中的 NA 值转换为 values 对象相应的 NA 值
    exp = _convert_na_value(values, exp)
    # 使用 pytest 的 assert_series_equal 函数比较 result 和 exp 是否相等
    tm.assert_series_equal(result, exp)


def test_split_blank_string(any_string_dtype):
    # 扩展空字符串分割 GH 20067
    # 创建一个包含多种类型对象的 Series，每个元素都是空字符串
    values = Series([""], name="test", dtype=any_string_dtype)
    # 调用 values 对象的 str 属性中的 split 方法，使用默认的空白字符分割，并扩展为 DataFrame
    result = values.str.split(expand=True)
    # 预期的结果是一个空的 DataFrame，注意：这不是一个空的 DataFrame，而是一个列为空的 DataFrame
    exp = DataFrame([[]], dtype=any_string_dtype)
    # 使用 pytest 的 assert_frame_equal 函数比较 result 和 exp 是否相等
    tm.assert_frame_equal(result, exp)


def test_split_blank_string_with_non_empty(any_string_dtype):
    # 创建一个包含多种类型对象的 Series，每个元素都是非空字符串
    values = Series(["a b c", "a b", "", " "], name="test", dtype=any_string_dtype)
    # 调用 values 对象的 str 属性中的 split 方法，使用默认的空白字符分割，并扩展为 DataFrame
    result = values.str.split(expand=True)
    # 预期的结果是一个包含各种对象类型的 DataFrame，其中字符串根据空白字符分割成列表，其他类型保持不变
    exp = DataFrame(
        [
            ["a", "b", "c"],
            ["a", "b", None],
            [None, None, None],
            [None, None, None],
        ],
        dtype=any_string_dtype,
    )
    # 使用 pytest 的 assert_frame_equal 函数比较 result 和 exp 是否相等
    tm.assert_frame_equal(result, exp)


@pytest.mark.parametrize("method", ["split", "rsplit"])
# 使用 pytest 的 parametrize 装饰器，为 test_split_noargs 函数参数化两种方法：split 和 rsplit
def test_split_noargs(any_string_dtype, method):
    # #1859
    # 创建一个包含多种类型对象的 Series，每个元素都是包含名字的字符串
    s = Series(["Wes McKinney", "Travis  Oliphant"], dtype=any_string_dtype)
    # 调用 s 对象的 str 属性中的 method 方法（split 或 rsplit），使用默认的空白字符分割
    result = getattr(s.str, method)()
    # 预期的结果是一个包含各种对象类型的 Series，其中字符串根据默认的空白字符分割成列表，保留最后一个名字部分
    expected = ["Travis", "Oliphant"]
    # 断言 result 中第二个元素是否等于 expected
    assert result[1] == expected


@pytest.mark.parametrize(
    "data, pat",
    [
        (["bd asdf jfg", "kjasdflqw asdfnfk"], None),
        (["bd asdf jfg", "kjasdflqw asdfnfk"], "asdf"),
        (["bd_asdf_jfg", "kjasdflqw_asdfnfk"], "_"),
    ],
)
@pytest.mark.parametrize("n", [-1, 0])
# 使用 pytest 的 parametrize 装饰器，为匿名的 test 函数参数化数据和模式，以及 n 参数的两种情况：-1 和 0
# 定义一个测试函数，用于测试字符串 Series 的 split 方法和相关功能
def test_split_maxsplit(data, pat, any_string_dtype, n):
    # 使用传入的数据和数据类型创建字符串 Series 对象 s
    s = Series(data, dtype=any_string_dtype)

    # 调用 Series 的 str.split 方法，根据给定的分隔符 pat 和最大分割次数 n 进行分割
    result = s.str.split(pat=pat, n=n)
    # 生成期望结果 xp，使用默认的分割方式（无限次分割）
    xp = s.str.split(pat=pat)
    # 使用 pytest 的断言方法验证 result 和 xp 是否相等
    tm.assert_series_equal(result, xp)


# 使用 pytest 的 parametrize 装饰器定义多组测试参数
@pytest.mark.parametrize(
    "data, pat, expected_val",
    [
        (
            ["split once", "split once too!"],
            None,
            "once too!",
        ),
        (
            ["split_once", "split_once_too!"],
            "_",
            "once_too!",
        ),
    ],
)
# 定义测试函数，用于验证在没有指定分割符 pat 但指定了非零分割次数 n 的情况下的 split 行为
def test_split_no_pat_with_nonzero_n(data, pat, expected_val, any_string_dtype):
    # 使用传入的数据和数据类型创建字符串 Series 对象 s
    s = Series(data, dtype=any_string_dtype)
    # 调用 Series 的 str.split 方法，指定分割次数为1
    result = s.str.split(pat=pat, n=1)
    # 生成期望结果 expected，手动创建 Series 对象，包含两个索引对应的列表
    expected = Series({0: ["split", "once"], 1: ["split", expected_val]})
    # 使用 pytest 的断言方法验证 result 和 expected 是否相等，忽略索引类型的检查
    tm.assert_series_equal(expected, result, check_index_type=False)


# 定义测试函数，用于验证不进行任何分割情况下的 split_to_dataframe 行为
def test_split_to_dataframe_no_splits(any_string_dtype):
    # 使用固定数据创建字符串 Series 对象 s
    s = Series(["nosplit", "alsonosplit"], dtype=any_string_dtype)
    # 调用 Series 的 str.split 方法，指定分隔符为 "_", 并扩展为 DataFrame
    result = s.str.split("_", expand=True)
    # 生成期望的 DataFrame 对象 exp，包含一个 Series 列
    exp = DataFrame({0: Series(["nosplit", "alsonosplit"], dtype=any_string_dtype)})
    # 使用 pytest 的断言方法验证 result 和 exp 是否相等
    tm.assert_frame_equal(result, exp)


# 定义测试函数，用于验证按指定分隔符进行 split_to_dataframe 的行为
def test_split_to_dataframe(any_string_dtype):
    # 使用固定数据创建字符串 Series 对象 s
    s = Series(["some_equal_splits", "with_no_nans"], dtype=any_string_dtype)
    # 调用 Series 的 str.split 方法，指定分隔符为 "_", 并扩展为 DataFrame
    result = s.str.split("_", expand=True)
    # 生成期望的 DataFrame 对象 exp，包含三个列，每列包含相应的拆分结果
    exp = DataFrame(
        {0: ["some", "with"], 1: ["equal", "no"], 2: ["splits", "nans"]},
        dtype=any_string_dtype,
    )
    # 使用 pytest 的断言方法验证 result 和 exp 是否相等
    tm.assert_frame_equal(result, exp)


# 定义测试函数，用于验证拆分结果长度不同情况下的 split_to_dataframe 行为
def test_split_to_dataframe_unequal_splits(any_string_dtype):
    # 使用固定数据创建字符串 Series 对象 s
    s = Series(
        ["some_unequal_splits", "one_of_these_things_is_not"], dtype=any_string_dtype
    )
    # 调用 Series 的 str.split 方法，指定分隔符为 "_", 并扩展为 DataFrame
    result = s.str.split("_", expand=True)
    # 生成期望的 DataFrame 对象 exp，包含六列，每列包含相应的拆分结果，空缺值填充为 None
    exp = DataFrame(
        {
            0: ["some", "one"],
            1: ["unequal", "of"],
            2: ["splits", "these"],
            3: [None, "things"],
            4: [None, "is"],
            5: [None, "not"],
        },
        dtype=any_string_dtype,
    )
    # 使用 pytest 的断言方法验证 result 和 exp 是否相等
    tm.assert_frame_equal(result, exp)


# 定义测试函数，用于验证拆分结果带有索引的 split_to_dataframe 行为
def test_split_to_dataframe_with_index(any_string_dtype):
    # 使用固定数据创建字符串 Series 对象 s，并指定索引
    s = Series(
        ["some_splits", "with_index"], index=["preserve", "me"], dtype=any_string_dtype
    )
    # 调用 Series 的 str.split 方法，指定分隔符为 "_", 并扩展为 DataFrame
    result = s.str.split("_", expand=True)
    # 生成期望的 DataFrame 对象 exp，包含两列，每列包含相应的拆分结果，并保留原始索引
    exp = DataFrame(
        {0: ["some", "with"], 1: ["splits", "index"]},
        index=["preserve", "me"],
        dtype=any_string_dtype,
    )
    # 使用 pytest 的断言方法验证 result 和 exp 是否相等
    tm.assert_frame_equal(result, exp)

    # 使用 pytest 的 raises 方法验证在指定非布尔型的 expand 参数时是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="expand must be"):
        s.str.split("_", expand="not_a_boolean")


# 定义测试函数，用于验证多级索引情况下的 split_to_multiindex_expand 行为
def test_split_to_multiindex_expand_no_splits():
    # 使用固定数据创建 Index 对象 idx
    idx = Index(["nosplit", "alsonosplit", np.nan])
    # 调用 Index 的 str.split 方法，指定分隔符为 "_", 并扩展为多级索引
    result = idx.str.split("_", expand=True)
    # 期望结果 exp 为原始 Index 对象 idx
    exp = idx
    # 使用 pytest 的断言方法验证 result 和 exp 是否相等
    tm.assert_index_equal(result, exp)
    # 验证 result 的级数是否为1
    assert result.nlevels == 1


# 定义测试函数，用于验证带有 NaN 和 None 的多级索引情况下的 split_to_multiindex_expand 行为
def test_split_to_multiindex_expand():
    # 使用固定数据创建 Index 对象 idx
    idx = Index(["some_equal_splits", "with_no_nans", np.nan, None])
    # 调用 Index 的 str.split 方法，指定分隔符为 "_", 并扩展为多级索引
    result = idx.str.split("_", expand=True)
    # 使用 MultiIndex 类的 from_tuples 方法创建一个多层索引对象 exp，
    # 包含了四个元组作为索引的各层，其中第三个元组包含了 NaN 值
    exp = MultiIndex.from_tuples(
        [
            ("some", "equal", "splits"),  # 第一个元组
            ("with", "no", "nans"),        # 第二个元组
            [np.nan, np.nan, np.nan],      # 第三个元组，包含 NaN 值
            [None, None, None],            # 第四个元组，包含 None 值
        ]
    )
    
    # 使用 tm.assert_index_equal 函数断言 result 和 exp 是相等的索引对象
    tm.assert_index_equal(result, exp)
    
    # 断言 result 对象的层级数（levels）为 3
    assert result.nlevels == 3
# 定义一个测试函数，用于测试字符串索引在使用 '_' 分割时的不等分割情况
def test_split_to_multiindex_expand_unequal_splits():
    # 创建一个包含字符串元素的索引对象
    idx = Index(["some_unequal_splits", "one_of_these_things_is_not", np.nan, None])
    # 对索引对象中的每个字符串元素按 '_' 进行分割，并扩展成DataFrame
    result = idx.str.split("_", expand=True)
    # 期望的多级索引对象，每个元组表示一个分割后的字符串
    exp = MultiIndex.from_tuples(
        [
            ("some", "unequal", "splits", np.nan, np.nan, np.nan),
            ("one", "of", "these", "things", "is", "not"),
            (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
            (None, None, None, None, None, None),
        ]
    )
    # 断言分割后的结果与期望的多级索引对象相等
    tm.assert_index_equal(result, exp)
    # 断言结果的层级数为6
    assert result.nlevels == 6

    # 使用pytest断言，测试非布尔型的expand参数会抛出值错误异常
    with pytest.raises(ValueError, match="expand must be"):
        idx.str.split("_", expand="not_a_boolean")


# 定义一个测试函数，测试字符串序列在使用 '_' 右向分割时不进行分割的情况
def test_rsplit_to_dataframe_expand_no_splits(any_string_dtype):
    # 创建一个字符串序列对象
    s = Series(["nosplit", "alsonosplit"], dtype=any_string_dtype)
    # 对序列对象中的每个字符串元素按 '_' 右向分割，并扩展成DataFrame
    result = s.str.rsplit("_", expand=True)
    # 期望的DataFrame对象，包含了原始字符串序列中的数据
    exp = DataFrame({0: Series(["nosplit", "alsonosplit"])}, dtype=any_string_dtype)
    # 断言分割后的结果DataFrame与期望的DataFrame相等
    tm.assert_frame_equal(result, exp)


# 定义一个测试函数，测试字符串序列在使用 '_' 右向分割时进行分割的情况
def test_rsplit_to_dataframe_expand(any_string_dtype):
    # 创建一个字符串序列对象
    s = Series(["some_equal_splits", "with_no_nans"], dtype=any_string_dtype)
    # 对序列对象中的每个字符串元素按 '_' 右向分割，并扩展成DataFrame
    result = s.str.rsplit("_", expand=True)
    # 期望的DataFrame对象，每列包含了分割后的字符串片段
    exp = DataFrame(
        {0: ["some", "with"], 1: ["equal", "no"], 2: ["splits", "nans"]},
        dtype=any_string_dtype,
    )
    # 断言分割后的结果DataFrame与期望的DataFrame相等
    tm.assert_frame_equal(result, exp)

    # 使用n参数限制分割的最大次数为2，再次断言分割结果与期望结果相等
    result = s.str.rsplit("_", expand=True, n=2)
    tm.assert_frame_equal(result, exp)

    # 使用n参数限制分割的最大次数为1，再次断言分割结果与期望结果相等
    result = s.str.rsplit("_", expand=True, n=1)
    exp = DataFrame(
        {0: ["some_equal", "with_no"], 1: ["splits", "nans"]}, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, exp)


# 定义一个测试函数，测试带索引的字符串序列在使用 '_' 右向分割时进行分割的情况
def test_rsplit_to_dataframe_expand_with_index(any_string_dtype):
    # 创建一个带有索引的字符串序列对象
    s = Series(
        ["some_splits", "with_index"], index=["preserve", "me"], dtype=any_string_dtype
    )
    # 对序列对象中的每个字符串元素按 '_' 右向分割，并扩展成DataFrame，保持索引
    result = s.str.rsplit("_", expand=True)
    # 期望的DataFrame对象，每列包含了分割后的字符串片段，保持原始的索引
    exp = DataFrame(
        {0: ["some", "with"], 1: ["splits", "index"]},
        index=["preserve", "me"],
        dtype=any_string_dtype,
    )
    # 断言分割后的结果DataFrame与期望的DataFrame相等
    tm.assert_frame_equal(result, exp)


# 定义一个测试函数，测试不进行分割的情况下对索引中的字符串进行右向分割
def test_rsplit_to_multiindex_expand_no_split():
    # 创建一个索引对象
    idx = Index(["nosplit", "alsonosplit"])
    # 对索引对象中的每个字符串元素按 '_' 右向分割，并扩展成MultiIndex
    result = idx.str.rsplit("_", expand=True)
    # 期望的MultiIndex对象，与原始索引对象相同，因为没有进行分割
    exp = idx
    # 断言分割后的结果MultiIndex与期望的MultiIndex相等
    tm.assert_index_equal(result, exp)
    # 断言结果的层级数为1
    assert result.nlevels == 1


# 定义一个测试函数，测试对索引中的字符串进行右向分割，并扩展成MultiIndex的情况
def test_rsplit_to_multiindex_expand():
    # 创建一个索引对象
    idx = Index(["some_equal_splits", "with_no_nans"])
    # 对索引对象中的每个字符串元素按 '_' 右向分割，并扩展成MultiIndex
    result = idx.str.rsplit("_", expand=True)
    # 期望的MultiIndex对象，每个元组表示一个分割后的字符串
    exp = MultiIndex.from_tuples([("some", "equal", "splits"), ("with", "no", "nans")])
    # 断言分割后的结果MultiIndex与期望的MultiIndex相等
    tm.assert_index_equal(result, exp)
    # 断言结果的层级数为3
    assert result.nlevels == 3


# 定义一个测试函数，测试对索引中的字符串进行右向分割，限制分割的最大次数为1，并扩展成MultiIndex的情况
def test_rsplit_to_multiindex_expand_n():
    # 创建一个索引对象
    idx = Index(["some_equal_splits", "with_no_nans"])
    # 对索引对象中的每个字符串元素按 '_' 右向分割，限制分割的最大次数为1，并扩展成MultiIndex
    result = idx.str.rsplit("_", expand=True, n=1)
    # 期望的MultiIndex对象，每个元组表示一个分割后的字符串
    exp = MultiIndex.from_tuples([("some_equal", "splits"), ("with_no", "nans")])
    # 断言分割后的结果MultiIndex与期望的MultiIndex相等
    tm.assert_index_equal(result, exp)
    # 断言结果的层级数为2
    assert result.nlevels == 2
def test_split_nan_expand(any_string_dtype):
    # 用例名称：test_split_nan_expand
    # GH 18450
    # 创建一个 Series 包含两个元素，一个是字符串 "foo,bar,baz"，另一个是 np.nan，数据类型为 any_string_dtype
    s = Series(["foo,bar,baz", np.nan], dtype=any_string_dtype)
    # 对 Series 中的每个字符串元素使用逗号分割，并扩展为 DataFrame
    result = s.str.split(",", expand=True)
    # 期望的结果是一个 DataFrame，包含 [["foo", "bar", "baz"], [np.nan, np.nan, np.nan]]，数据类型为 any_string_dtype
    exp = DataFrame(
        [["foo", "bar", "baz"], [np.nan, np.nan, np.nan]], dtype=any_string_dtype
    )
    # 断言 result 与 exp 相等
    tm.assert_frame_equal(result, exp)

    # 检查结果中的值确实是 np.nan 或 pd.NA，而不是 None
    # TODO 见 GH 18463
    # tm.assert_frame_equal 无法区分它们
    if any_string_dtype in object_pyarrow_numpy:
        # 如果数据类型是 object、pyarrow 或 numpy，则断言 result.iloc[1] 中的所有值都是 np.nan
        assert all(np.isnan(x) for x in result.iloc[1])
    else:
        # 否则，断言 result.iloc[1] 中的所有值都是 pd.NA
        assert all(x is pd.NA for x in result.iloc[1])


def test_split_with_name_series(any_string_dtype):
    # 用例名称：test_split_with_name_series
    # GH 12617

    # 应保留名称的 Series
    # 创建一个 Series，包含两个元素 "a,b" 和 "c,d"，名称为 "xxx"，数据类型为 any_string_dtype
    s = Series(["a,b", "c,d"], name="xxx", dtype=any_string_dtype)
    # 对 Series 中的每个字符串元素使用逗号分割
    res = s.str.split(",")
    # 期望的结果是一个 Series，包含 [["a", "b"], ["c", "d"]]，名称为 "xxx"
    exp = Series([["a", "b"], ["c", "d"]], name="xxx")
    # 断言 res 与 exp 相等
    tm.assert_series_equal(res, exp)

    # 对 Series 中的每个字符串元素使用逗号分割，并扩展为 DataFrame
    res = s.str.split(",", expand=True)
    # 期望的结果是一个 DataFrame，包含 [["a", "b"], ["c", "d"]]，数据类型为 any_string_dtype
    exp = DataFrame([["a", "b"], ["c", "d"]], dtype=any_string_dtype)
    # 断言 res 与 exp 相等
    tm.assert_frame_equal(res, exp)


def test_split_with_name_index():
    # 用例名称：test_split_with_name_index
    # GH 12617
    # 创建一个 Index，包含两个元素 "a,b" 和 "c,d"，名称为 "xxx"
    idx = Index(["a,b", "c,d"], name="xxx")
    # 对 Index 中的每个字符串元素使用逗号分割
    res = idx.str.split(",")
    # 期望的结果是一个 Index，包含 [["a", "b"], ["c", "d"]]，名称为 "xxx"
    exp = Index([["a", "b"], ["c", "d"]], name="xxx")
    # 断言 res 与 exp 相等
    tm.assert_index_equal(res, exp)

    # 对 Index 中的每个字符串元素使用逗号分割，并扩展为 MultiIndex
    res = idx.str.split(",", expand=True)
    # 期望的结果是一个 MultiIndex，包含 [("a", "b"), ("c", "d")]，数据类型为 any_string_dtype
    exp = MultiIndex.from_tuples([("a", "b"), ("c", "d")])
    # 断言 res 的层级数为 2
    assert res.nlevels == 2
    # 断言 res 与 exp 相等
    tm.assert_index_equal(res, exp)


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            [
                ("a", "__", "b__c"),
                ("c", "__", "d__e"),
                np.nan,
                ("f", "__", "g__h"),
                None,
            ],
        ],
        [
            "rpartition",
            [
                ("a__b", "__", "c"),
                ("c__d", "__", "e"),
                np.nan,
                ("f__g", "__", "h"),
                None,
            ],
        ],
    ],
)
def test_partition_series_more_than_one_char(method, exp, any_string_dtype):
    # 用例名称：test_partition_series_more_than_one_char
    # https://github.com/pandas-dev/pandas/issues/23558
    # 字符串长度超过一个字符的情况
    # 创建一个 Series，包含五个元素 "a__b__c", "c__d__e", np.nan, "f__g__h", None，数据类型为 any_string_dtype
    s = Series(["a__b__c", "c__d__e", np.nan, "f__g__h", None], dtype=any_string_dtype)
    # 调用 Series 的 partition 或 rpartition 方法，使用 "__" 分割字符串，不扩展
    result = getattr(s.str, method)("__", expand=False)
    # 期望的结果是一个 Series，包含 exp 中定义的元组序列，经过 _convert_na_value 处理后
    expected = Series(exp)
    expected = _convert_na_value(s, expected)
    # 断言 result 与 expected 相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            [("a", " ", "b c"), ("c", " ", "d e"), np.nan, ("f", " ", "g h"), None],
        ],
        [
            "rpartition",
            [("a b", " ", "c"), ("c d", " ", "e"), np.nan, ("f g", " ", "h"), None],
        ],
    ],
)
def test_partition_series_none(any_string_dtype, method, exp):
    # 用例名称：test_partition_series_none
    # https://github.com/pandas-dev/pandas/issues/23558
    # None 的情况
    # 创建一个 Series，包含五个元素 "a b c", "c d e", np.nan, "f g h", None，数据类型为 any_string_dtype
    s = Series(["a b c", "c d e", np.nan, "f g h", None], dtype=any_string_dtype)
    # 调用 Series 的 partition 或 rpartition 方法，使用 " " 分割字符串，不扩展
    result = getattr(s.str, method)(expand=False)
    # 创建一个名为 expected 的 Series 对象，使用变量 exp 中的数据
    expected = Series(exp)
    
    # 调用 _convert_na_value 函数，处理 Series 对象 s 和 expected，处理缺失值的转换
    expected = _convert_na_value(s, expected)
    
    # 使用 tm.assert_series_equal 函数比较 result 和 expected 两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "method, exp",
    [  # 参数化测试数据，每个元素是一个方法名和期望结果的元组列表
        [
            "partition",  # 使用字符串分割方法 partition
            [("abc", "", ""), ("cde", "", ""), np.nan, ("fgh", "", ""), None],  # 预期的分割结果列表
        ],
        [
            "rpartition",  # 使用字符串反向分割方法 rpartition
            [("", "", "abc"), ("", "", "cde"), np.nan, ("", "", "fgh"), None],  # 预期的反向分割结果列表
        ],
    ],
)
def test_partition_series_not_split(any_string_dtype, method, exp):
    # https://github.com/pandas-dev/pandas/issues/23558
    # Not split
    s = Series(["abc", "cde", np.nan, "fgh", None], dtype=any_string_dtype)  # 创建包含字符串的 Series 对象
    result = getattr(s.str, method)("_", expand=False)  # 调用字符串的指定方法，并指定分割字符 "_"
    expected = Series(exp)  # 创建期望的结果 Series 对象
    expected = _convert_na_value(s, expected)  # 处理 NaN 值的辅助函数调用
    tm.assert_series_equal(result, expected)  # 使用测试工具比较实际结果和期望结果的 Series 对象


@pytest.mark.parametrize(
    "method, exp",
    [  # 参数化测试数据，每个元素是一个方法名和期望结果的元组列表
        [
            "partition",  # 使用字符串分割方法 partition
            [("a", "_", "b_c"), ("c", "_", "d_e"), np.nan, ("f", "_", "g_h")],  # 预期的分割结果列表
        ],
        [
            "rpartition",  # 使用字符串反向分割方法 rpartition
            [("a_b", "_", "c"), ("c_d", "_", "e"), np.nan, ("f_g", "_", "h")],  # 预期的反向分割结果列表
        ],
    ],
)
def test_partition_series_unicode(any_string_dtype, method, exp):
    # https://github.com/pandas-dev/pandas/issues/23558
    # unicode
    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)  # 创建包含 Unicode 字符串的 Series 对象
    result = getattr(s.str, method)("_", expand=False)  # 调用字符串的指定方法，并指定分割字符 "_"
    expected = Series(exp)  # 创建期望的结果 Series 对象
    expected = _convert_na_value(s, expected)  # 处理 NaN 值的辅助函数调用
    tm.assert_series_equal(result, expected)  # 使用测试工具比较实际结果和期望结果的 Series 对象


@pytest.mark.parametrize("method", ["partition", "rpartition"])
def test_partition_series_stdlib(any_string_dtype, method):
    # https://github.com/pandas-dev/pandas/issues/23558
    # compare to standard lib
    s = Series(["A_B_C", "B_C_D", "E_F_G", "EFGHEF"], dtype=any_string_dtype)  # 创建包含标准库中字符串的 Series 对象
    result = getattr(s.str, method)("_", expand=False).tolist()  # 调用字符串的指定方法，并指定分割字符 "_"，转换为列表
    assert result == [getattr(v, method)("_") for v in s]  # 使用标准库中字符串对象进行比较


@pytest.mark.parametrize(
    "method, exp",
    [  # 参数化测试数据，每个元素是一个方法名和期望结果的元组列表
        [
            "partition",  # 使用字符串分割方法 partition
            [("a", "_", "b_c"), ("c", "_", "d_e"), ("f", "_", "g_h"), np.nan, None],  # 预期的分割结果列表
        ],
        [
            "rpartition",  # 使用字符串反向分割方法 rpartition
            [("a_b", "_", "c"), ("c_d", "_", "e"), ("f_g", "_", "h"), np.nan, None],  # 预期的反向分割结果列表
        ],
    ],
)
def test_partition_index(method, exp):
    # https://github.com/pandas-dev/pandas/issues/23558
    values = Index(["a_b_c", "c_d_e", "f_g_h", np.nan, None])  # 创建包含索引值的 Index 对象
    result = getattr(values.str, method)("_", expand=False)  # 调用字符串的指定方法，并指定分割字符 "_"
    exp = Index(np.array(exp, dtype=object), dtype=object)  # 创建期望的索引结果对象
    tm.assert_index_equal(result, exp)  # 使用测试工具比较实际结果和期望结果的 Index 对象
    assert result.nlevels == 1  # 断言实际结果的层级数为 1


@pytest.mark.parametrize(
    "method, exp",
    [  # 参数化测试数据，每个元素是一个方法名和期望结果的字典列表
        [
            "partition",  # 使用字符串分割方法 partition
            {  # 预期的分割结果字典，键是分割结果的位置，值是对应位置的列表
                0: ["a", "c", np.nan, "f", None],
                1: ["_", "_", np.nan, "_", None],
                2: ["b_c", "d_e", np.nan, "g_h", None],
            },
        ],
        [
            "rpartition",  # 使用字符串反向分割方法 rpartition
            {  # 预期的反向分割结果字典，键是分割结果的位置，值是对应位置的列表
                0: ["a_b", "c_d", np.nan, "f_g", None],
                1: ["_", "_", np.nan, "_", None],
                2: ["c", "e", np.nan, "h", None],
            },
        ],
    ],



# 这行代码是一个列表的结尾，闭合了一个包含多个元素的列表
def test_partition_to_dataframe(any_string_dtype, method, exp):
    # 函数用于测试将字符串序列进行分区操作后转换为DataFrame的功能
    # 参考：https://github.com/pandas-dev/pandas/issues/23558
    
    # 创建测试用的字符串序列
    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h", None], dtype=any_string_dtype)
    
    # 调用字符串序列的指定方法（partition或rpartition）进行分区操作
    result = getattr(s.str, method)("_")
    
    # 创建期望的DataFrame，用于与结果进行比较
    expected = DataFrame(
        exp,
        dtype=any_string_dtype,
    )
    
    # 使用pandas的测试工具进行DataFrame的比较
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            {
                0: ["a", "c", np.nan, "f", None],
                1: ["_", "_", np.nan, "_", None],
                2: ["b_c", "d_e", np.nan, "g_h", None],
            },
        ],
        [
            "rpartition",
            {
                0: ["a_b", "c_d", np.nan, "f_g", None],
                1: ["_", "_", np.nan, "_", None],
                2: ["c", "e", np.nan, "h", None],
            },
        ],
    ],
)
def test_partition_to_dataframe_from_series(any_string_dtype, method, exp):
    # 函数用于测试从字符串序列中生成DataFrame，使用分区方法（partition或rpartition）
    # 参考：https://github.com/pandas-dev/pandas/issues/23558
    
    # 创建测试用的字符串序列
    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h", None], dtype=any_string_dtype)
    
    # 调用字符串序列的指定方法（partition或rpartition），并扩展结果为DataFrame
    result = getattr(s.str, method)("_", expand=True)
    
    # 创建期望的DataFrame，用于与结果进行比较
    expected = DataFrame(
        exp,
        dtype=any_string_dtype,
    )
    
    # 使用pandas的测试工具进行DataFrame的比较
    tm.assert_frame_equal(result, expected)


def test_partition_with_name(any_string_dtype):
    # 函数用于测试在指定名称下进行字符串分区操作后转换为DataFrame的功能
    # 参考：GH 12617
    
    # 创建测试用的字符串序列，指定名称为"xxx"
    s = Series(["a,b", "c,d"], name="xxx", dtype=any_string_dtype)
    
    # 调用字符串序列的分区方法（partition），并将结果转换为DataFrame
    result = s.str.partition(",")
    
    # 创建期望的DataFrame，包含分区后的三个部分
    expected = DataFrame(
        {0: ["a", "c"], 1: [",", ","], 2: ["b", "d"]}, dtype=any_string_dtype
    )
    
    # 使用pandas的测试工具进行DataFrame的比较
    tm.assert_frame_equal(result, expected)


def test_partition_with_name_expand(any_string_dtype):
    # 函数用于测试在指定名称下进行字符串分区操作，并保留名称的功能
    # 参考：GH 12617
    
    # 创建测试用的字符串序列，指定名称为"xxx"
    s = Series(["a,b", "c,d"], name="xxx", dtype=any_string_dtype)
    
    # 调用字符串序列的分区方法（partition），并扩展结果为Series，保留名称
    result = s.str.partition(",", expand=False)
    
    # 创建期望的Series，包含分区后的三个部分
    expected = Series([("a", ",", "b"), ("c", ",", "d")], name="xxx")
    
    # 使用pandas的测试工具进行Series的比较
    tm.assert_series_equal(result, expected)


def test_partition_index_with_name():
    # 函数用于测试在索引对象中进行字符串分区操作后转换为MultiIndex的功能
    # 参考：GH 12617
    
    # 创建测试用的索引对象，指定名称为"xxx"
    idx = Index(["a,b", "c,d"], name="xxx")
    
    # 调用索引对象的分区方法（partition），并将结果转换为MultiIndex
    result = idx.str.partition(",")
    
    # 创建期望的MultiIndex，包含分区后的三个部分
    expected = MultiIndex.from_tuples([("a", ",", "b"), ("c", ",", "d")])
    
    # 断言MultiIndex的层级数
    assert result.nlevels == 3
    
    # 使用pandas的测试工具进行MultiIndex的比较
    tm.assert_index_equal(result, expected)


def test_partition_index_with_name_expand_false():
    # 函数用于测试在索引对象中进行字符串分区操作，并保留名称的功能
    # 参考：GH 12617
    
    # 创建测试用的索引对象，指定名称为"xxx"
    idx = Index(["a,b", "c,d"], name="xxx")
    
    # 调用索引对象的分区方法（partition），并扩展结果为Index，保留名称
    result = idx.str.partition(",", expand=False)
    
    # 创建期望的Index，包含分区后的三个部分
    expected = Index(np.array([("a", ",", "b"), ("c", ",", "d")]), name="xxx")
    
    # 断言Index的层级数
    assert result.nlevels == 1
    
    # 使用pandas的测试工具进行Index的比较
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("method", ["partition", "rpartition"])
def test_partition_sep_kwarg(any_string_dtype, method):
    # 函数用于测试分区方法（partition或rpartition）中的关键字参数"sep"的功能
    # 参考：GH 22676; depr kwarg "pat" in favor of "sep"
    
    # 创建测试用的字符串序列
    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)
    
    # 分别使用关键字参数"sep"和直接传递分隔符来调用分区方法（partition或rpartition）
    expected = getattr(s.str, method)(sep="_")
    result = getattr(s.str, method)("_")
    
    # 使用pandas的测试工具进行DataFrame的比较
    tm.assert_frame_equal(result, expected)


def test_get():
    # 函数用于测试获取字符串序列中的元素
    ser = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"])
    # 使用 Pandas 的 Series 对象 ser 进行操作，将每个元素按照 "_" 分割后取第二部分，存入 result
    result = ser.str.split("_").str.get(1)
    
    # 创建一个预期的 Series 对象 expected，包含字符串 "b", "d", np.nan, "g"，数据类型为 object
    expected = Series(["b", "d", np.nan, "g"], dtype=object)
    
    # 使用 Pandas 测试模块 tm 来比较 result 和 expected 两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试从混合对象中获取指定位置的子项
def test_get_mixed_object():
    # 创建一个 Series 对象，包含多种类型的数据
    ser = Series(["a_b_c", np.nan, "c_d_e", True, datetime.today(), None, 1, 2.0])
    # 对 Series 中的每个字符串进行按 '_' 分割，并获取分割结果的第二个元素
    result = ser.str.split("_").str.get(1)
    # 创建预期的 Series 对象，包含对应位置的预期结果
    expected = Series(
        ["b", np.nan, "d", np.nan, np.nan, None, np.nan, np.nan], dtype=object
    )
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


# 使用参数化测试，对指定位置的字符串进行分割和获取操作，检查边界情况
@pytest.mark.parametrize("idx", [2, -3])
def test_get_bounds(idx):
    # 创建一个 Series 对象，包含多个用 '_' 分隔的字符串
    ser = Series(["1_2_3_4_5", "6_7_8_9_10", "11_12"])
    # 对 Series 中的每个字符串进行按 '_' 分割，并获取指定位置的元素
    result = ser.str.split("_").str.get(idx)
    # 创建预期的 Series 对象，包含对应位置的预期结果
    expected = Series(["3", "8", np.nan], dtype=object)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


# 使用参数化测试，对复杂的数据类型进行获取操作，验证是否正确处理不在字典中的键引发的 KeyError
@pytest.mark.parametrize(
    "idx, exp", [[2, [3, 3, np.nan, "b"]], [-1, [3, 3, np.nan, np.nan]]]
)
def test_get_complex(idx, exp):
    # 创建一个 Series 对象，包含多种复杂的数据类型
    ser = Series([(1, 2, 3), [1, 2, 3], {1, 2, 3}, {1: "a", 2: "b", 3: "c"}])
    # 对 Series 中的每个元素进行获取操作，获取指定位置的子项
    result = ser.str.get(idx)
    # 创建预期的 Series 对象，包含对应位置的预期结果
    expected = Series(exp)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


# 使用参数化测试，对复杂嵌套的数据类型进行获取操作，验证是否正确处理
@pytest.mark.parametrize("to_type", [tuple, list, np.array])
def test_get_complex_nested(to_type):
    # 创建一个 Series 对象，包含复杂嵌套的数据类型
    ser = Series([to_type([to_type([1, 2])])])
    # 对 Series 中的每个元素进行获取操作，获取指定位置的子项
    result = ser.str.get(0)
    # 创建预期的 Series 对象，包含对应位置的预期结果
    expected = Series([to_type([1, 2])])
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)

    # 对 Series 中的每个元素进行获取操作，获取超出索引范围的子项
    result = ser.str.get(1)
    # 创建预期的 Series 对象，包含对应位置的预期结果
    expected = Series([np.nan])
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


# 测试从字符串序列中获取指定位置的字符
def test_get_strings(any_string_dtype):
    # 创建一个 Series 对象，包含多个字符串
    ser = Series(["a", "ab", np.nan, "abc"], dtype=any_string_dtype)
    # 对 Series 中的每个字符串进行获取操作，获取指定位置的字符
    result = ser.str.get(2)
    # 创建预期的 Series 对象，包含对应位置的预期结果
    expected = Series([np.nan, np.nan, np.nan, "c"], dtype=any_string_dtype)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
```