# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_constructors.py`

```
# 导入所需的模块和库
from datetime import (
    date,  # 导入日期对象
    datetime,  # 导入日期时间对象
)
import itertools  # 导入 itertools 模块

import numpy as np  # 导入 NumPy 库并使用别名 np
import pytest  # 导入 pytest 测试框架

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike  # 从 pandas 中导入数据类型转换函数

import pandas as pd  # 导入 pandas 库并使用别名 pd
from pandas import (
    Index,  # 从 pandas 中导入 Index 类
    MultiIndex,  # 从 pandas 中导入 MultiIndex 类
    Series,  # 从 pandas 中导入 Series 类
    Timestamp,  # 从 pandas 中导入 Timestamp 类
    date_range,  # 从 pandas 中导入 date_range 函数
)
import pandas._testing as tm  # 导入 pandas 内部测试模块


def test_constructor_single_level():
    # 测试单层 MultiIndex 的构造函数
    result = MultiIndex(
        levels=[["foo", "bar", "baz", "qux"]], codes=[[0, 1, 2, 3]], names=["first"]
    )
    assert isinstance(result, MultiIndex)  # 断言 result 是 MultiIndex 类的实例
    expected = Index(["foo", "bar", "baz", "qux"], name="first")
    tm.assert_index_equal(result.levels[0], expected)  # 使用 pandas 内部测试函数验证 levels 的正确性
    assert result.names == ["first"]  # 断言 result 的 names 属性为 ["first"]


def test_constructor_no_levels():
    # 测试没有指定 levels 的 MultiIndex 构造函数
    msg = "non-zero number of levels/codes"
    with pytest.raises(ValueError, match=msg):
        MultiIndex(levels=[], codes=[])  # 期望抛出 ValueError 异常，错误信息包含 "non-zero number of levels/codes"

    msg = "Must pass both levels and codes"
    with pytest.raises(TypeError, match=msg):
        MultiIndex(levels=[])  # 期望抛出 TypeError 异常，错误信息包含 "Must pass both levels and codes"
    with pytest.raises(TypeError, match=msg):
        MultiIndex(codes=[])  # 期望抛出 TypeError 异常，错误信息包含 "Must pass both levels and codes"


def test_constructor_nonhashable_names():
    # 测试 MultiIndex 构造函数中不可哈希类型的 names
    levels = [[1, 2], ["one", "two"]]
    codes = [[0, 0, 1, 1], [0, 1, 0, 1]]
    names = (["foo"], ["bar"])
    msg = r"MultiIndex\.name must be a hashable type"
    with pytest.raises(TypeError, match=msg):
        MultiIndex(levels=levels, codes=codes, names=names)  # 期望抛出 TypeError 异常，错误信息符合正则表达式 msg

    # 使用 .rename() 方法测试
    mi = MultiIndex(
        levels=[[1, 2], ["one", "two"]],
        codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
        names=("foo", "bar"),
    )
    renamed = [["fooo"], ["barr"]]
    with pytest.raises(TypeError, match=msg):
        mi.rename(names=renamed)  # 期望抛出 TypeError 异常，错误信息符合正则表达式 msg

    # 使用 .set_names() 方法测试
    with pytest.raises(TypeError, match=msg):
        mi.set_names(names=renamed)  # 期望抛出 TypeError 异常，错误信息符合正则表达式 msg


def test_constructor_mismatched_codes_levels(idx):
    codes = [np.array([1]), np.array([2]), np.array([3])]
    levels = ["a"]

    msg = "Length of levels and codes must be the same"
    with pytest.raises(ValueError, match=msg):
        MultiIndex(levels=levels, codes=codes)  # 期望抛出 ValueError 异常，错误信息包含 "Length of levels and codes must be the same"

    length_error = (
        r"On level 0, code max \(3\) >= length of level \(1\)\. "
        "NOTE: this index is in an inconsistent state"
    )
    label_error = r"Unequal code lengths: \[4, 2\]"
    code_value_error = r"On level 0, code value \(-2\) < -1"

    # 对不同情况进行异常检查
    with pytest.raises(ValueError, match=length_error):
        MultiIndex(levels=[["a"], ["b"]], codes=[[0, 1, 2, 3], [0, 3, 4, 1]])  # 期望抛出 ValueError 异常，错误信息符合 length_error

    with pytest.raises(ValueError, match=label_error):
        MultiIndex(levels=[["a"], ["b"]], codes=[[0, 0, 0, 0], [0, 0]])  # 期望抛出 ValueError 异常，错误信息符合 label_error

    # 外部 API 测试
    with pytest.raises(ValueError, match=length_error):
        idx.copy().set_levels([["a"], ["b"]])  # 期望抛出 ValueError 异常，错误信息符合 length_error

    with pytest.raises(ValueError, match=label_error):
        idx.copy().set_codes([[0, 0, 0, 0], [0, 0]])  # 期望抛出 ValueError 异常，错误信息符合 label_error

    # 使用 verify_integrity=False 测试 set_codes 方法，不应该抛出任何异常
    # 复制索引并设置代码，此处设置了新的代码数组
    idx.copy().set_codes(codes=[[0, 0, 0, 0], [0, 0]], verify_integrity=False)

    # 当代码值小于 -1 时，验证引发 ValueError 异常，匹配错误消息为 code_value_error
    with pytest.raises(ValueError, match=code_value_error):
        # 创建一个 MultiIndex 对象，传入指定的 levels 和 codes 参数
        MultiIndex(levels=[["a"], ["b"]], codes=[[0, -2], [0, 0]])
# 定义一个测试函数，用于验证处理缺失值（NaN, NaT, None）时，MultiIndex 对象的行为是否正确
def test_na_levels():
    # GH26408
    # 测试：当有缺失值（NaN, NaT, None）时，是否重新分配代码值为-1
    result = MultiIndex(
        levels=[[np.nan, None, pd.NaT, 128, 2]], codes=[[0, -1, 1, 2, 3, 4]]
    )
    expected = MultiIndex(
        levels=[[np.nan, None, pd.NaT, 128, 2]], codes=[[-1, -1, -1, -1, 3, 4]]
    )
    tm.assert_index_equal(result, expected)

    result = MultiIndex(
        levels=[[np.nan, "s", pd.NaT, 128, None]], codes=[[0, -1, 1, 2, 3, 4]]
    )
    expected = MultiIndex(
        levels=[[np.nan, "s", pd.NaT, 128, None]], codes=[[-1, -1, 1, -1, 3, -1]]
    )
    tm.assert_index_equal(result, expected)

    # 验证 set_levels 和 set_codes 方法
    result = MultiIndex(
        levels=[[1, 2, 3, 4, 5]], codes=[[0, -1, 1, 2, 3, 4]]
    ).set_levels([[np.nan, "s", pd.NaT, 128, None]])
    tm.assert_index_equal(result, expected)

    result = MultiIndex(
        levels=[[np.nan, "s", pd.NaT, 128, None]], codes=[[1, 2, 2, 2, 2, 2]]
    ).set_codes([[0, -1, 1, 2, 3, 4]])
    tm.assert_index_equal(result, expected)


# ----------------------------------------------------------------------------
# from_arrays
# ----------------------------------------------------------------------------
# 使用 from_arrays 函数创建 MultiIndex 对象的测试函数
def test_from_arrays(idx):
    # 从 idx.levels 和 idx.codes 中创建数组，作为 from_arrays 的输入
    arrays = [
        np.asarray(lev).take(level_codes)
        for lev, level_codes in zip(idx.levels, idx.codes)
    ]

    # 使用数组列表作为输入创建 MultiIndex 对象，并验证结果是否与 idx 相等
    result = MultiIndex.from_arrays(arrays, names=idx.names)
    tm.assert_index_equal(result, idx)

    # 正确推断输入类型
    result = MultiIndex.from_arrays([[pd.NaT, Timestamp("20130101")], ["a", "b"]])
    assert result.levels[0].equals(Index([Timestamp("20130101")]))
    assert result.levels[1].equals(Index(["a", "b"]))


# 使用迭代器作为 from_arrays 输入的测试函数
def test_from_arrays_iterator(idx):
    # 从 idx.levels 和 idx.codes 中创建数组，作为 from_arrays 的输入
    arrays = [
        np.asarray(lev).take(level_codes)
        for lev, level_codes in zip(idx.levels, idx.codes)
    ]

    # 使用迭代器作为输入创建 MultiIndex 对象，并验证结果是否与 idx 相等
    result = MultiIndex.from_arrays(iter(arrays), names=idx.names)
    tm.assert_index_equal(result, idx)

    # 错误的迭代器输入应该引发 TypeError
    msg = "Input must be a list / sequence of array-likes."
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_arrays(0)


# 使用元组的元组作为 from_arrays 输入的测试函数
def test_from_arrays_tuples(idx):
    # 从 idx.levels 和 idx.codes 中创建数组，作为 from_arrays 的输入
    arrays = tuple(
        tuple(np.asarray(lev).take(level_codes))
        for lev, level_codes in zip(idx.levels, idx.codes)
    )

    # 使用元组的元组作为输入创建 MultiIndex 对象，并验证结果是否与 idx 相等
    result = MultiIndex.from_arrays(arrays, names=idx.names)
    tm.assert_index_equal(result, idx)
    # 创建一个包含多个元组的列表，每个元组包含两个 Pandas 时间序列对象
    [
        (
            # 创建一个从 "2011-01-01" 开始，每天一个周期，共3个周期的日期周期对象
            pd.period_range("2011-01-01", freq="D", periods=3),
            # 创建一个从 "2015-01-01" 开始，每小时一个周期，共3个周期的日期周期对象
            pd.period_range("2015-01-01", freq="h", periods=3),
        ),
        (
            # 创建一个从 "2015-01-01 10:00" 开始，每天一个周期，共3个周期的带时区的日期对象
            date_range("2015-01-01 10:00", freq="D", periods=3, tz="US/Eastern"),
            # 创建一个从 "2015-01-01 10:00" 开始，每小时一个周期，共3个周期的带时区的日期对象
            date_range("2015-01-01 10:00", freq="h", periods=3, tz="Asia/Tokyo"),
        ),
        (
            # 创建一个时间增量对象列表，从 "1 days" 开始，每天一个周期，共3个周期
            pd.timedelta_range("1 days", freq="D", periods=3),
            # 创建一个时间增量对象列表，从 "2 hours" 开始，每小时一个周期，共3个周期
            pd.timedelta_range("2 hours", freq="h", periods=3),
        ),
    ],
)
def test_from_arrays_index_series_period_datetimetz_and_timedelta(idx1, idx2):
    # 创建 MultiIndex 对象，使用给定的 idx1 和 idx2 数组作为级别
    result = MultiIndex.from_arrays([idx1, idx2])
    # 断言第一个级别的值与 idx1 相等
    tm.assert_index_equal(result.get_level_values(0), idx1)
    # 断言第二个级别的值与 idx2 相等
    tm.assert_index_equal(result.get_level_values(1), idx2)

    # 创建另一个 MultiIndex 对象，使用包含 idx1 和 idx2 的 Series 对象作为级别
    result2 = MultiIndex.from_arrays([Series(idx1), Series(idx2)])
    # 断言第一个级别的值与 idx1 相等
    tm.assert_index_equal(result2.get_level_values(0), idx1)
    # 断言第二个级别的值与 idx2 相等
    tm.assert_index_equal(result2.get_level_values(1), idx2)

    # 断言两个 MultiIndex 对象相等
    tm.assert_index_equal(result, result2)


def test_from_arrays_index_datetimelike_mixed():
    # 创建包含日期范围的 idx1，时钟频率的 idx2，时间差范围的 idx3 和周期范围的 idx4
    idx1 = date_range("2015-01-01 10:00", freq="D", periods=3, tz="US/Eastern")
    idx2 = date_range("2015-01-01 10:00", freq="h", periods=3)
    idx3 = pd.timedelta_range("1 days", freq="D", periods=3)
    idx4 = pd.period_range("2011-01-01", freq="D", periods=3)

    # 创建 MultiIndex 对象，使用给定的 idx1, idx2, idx3 和 idx4 数组作为级别
    result = MultiIndex.from_arrays([idx1, idx2, idx3, idx4])
    # 断言各级别的值与对应的 idx 数组相等
    tm.assert_index_equal(result.get_level_values(0), idx1)
    tm.assert_index_equal(result.get_level_values(1), idx2)
    tm.assert_index_equal(result.get_level_values(2), idx3)
    tm.assert_index_equal(result.get_level_values(3), idx4)

    # 创建另一个 MultiIndex 对象，使用包含 idx1, idx2, idx3 和 idx4 的 Series 对象作为级别
    result2 = MultiIndex.from_arrays(
        [Series(idx1), Series(idx2), Series(idx3), Series(idx4)]
    )
    # 断言各级别的值与对应的 idx 数组相等
    tm.assert_index_equal(result2.get_level_values(0), idx1)
    tm.assert_index_equal(result2.get_level_values(1), idx2)
    tm.assert_index_equal(result2.get_level_values(2), idx3)
    tm.assert_index_equal(result2.get_level_values(3), idx4)

    # 断言两个 MultiIndex 对象相等
    tm.assert_index_equal(result, result2)


def test_from_arrays_index_series_categorical():
    # 创建分类索引 idx1 和 idx2，用于测试 MultiIndex 的创建
    # GH13743 表明该部分代码是为了解决特定的问题或支持特定的功能
    idx1 = pd.CategoricalIndex(list("abcaab"), categories=list("bac"), ordered=False)
    idx2 = pd.CategoricalIndex(list("abcaab"), categories=list("bac"), ordered=True)

    # 创建 MultiIndex 对象，使用给定的 idx1 和 idx2 数组作为级别
    result = MultiIndex.from_arrays([idx1, idx2])
    # 断言第一个级别的值与 idx1 相等
    tm.assert_index_equal(result.get_level_values(0), idx1)
    # 断言第二个级别的值与 idx2 相等
    tm.assert_index_equal(result.get_level_values(1), idx2)

    # 创建另一个 MultiIndex 对象，使用包含 idx1 和 idx2 的 Series 对象作为级别
    result2 = MultiIndex.from_arrays([Series(idx1), Series(idx2)])
    # 断言第一个级别的值与 idx1 相等
    tm.assert_index_equal(result2.get_level_values(0), idx1)
    # 断言第二个级别的值与 idx2 相等
    tm.assert_index_equal(result2.get_level_values(1), idx2)

    # 创建另一个 MultiIndex 对象，使用 idx1.values 和 idx2.values 作为级别
    result3 = MultiIndex.from_arrays([idx1.values, idx2.values])
    # 断言第一个级别的值与 idx1 相等
    tm.assert_index_equal(result3.get_level_values(0), idx1)
    # 断言第二个级别的值与 idx2 相等
    tm.assert_index_equal(result3.get_level_values(1), idx2)


def test_from_arrays_empty():
    # 测试空数组作为输入时的行为
    # 0 levels
    msg = "Must pass non-zero number of levels/codes"
    # 使用 pytest 引发 ValueError 异常，检查是否包含指定的错误信息
    with pytest.raises(ValueError, match=msg):
        MultiIndex.from_arrays(arrays=[])

    # 1 level
    # 创建包含一个空数组的 MultiIndex 对象，命名为 "A"
    result = MultiIndex.from_arrays(arrays=[[]], names=["A"])
    # 断言 result 是 MultiIndex 类型的对象
    assert isinstance(result, MultiIndex)
    # 创建预期的空索引对象
    expected = Index([], name="A")
    # 断言 MultiIndex 对象的第一个级别与预期的空索引对象相等
    tm.assert_index_equal(result.levels[0], expected)
    # 断言 MultiIndex 对象的名称与预期名称列表相等
    assert result.names == ["A"]

    # N levels
    # 对于给定的每个 N 值进行循环，N 分别为 2 和 3
    for N in [2, 3]:
        # 创建包含 N 个空列表的列表，注意这里使用了乘法操作符复制列表，但所有子列表实际上是同一个对象的引用
        arrays = [[]] * N
        # 从字符串 "ABC" 中取前 N 个字符作为列表，例如当 N=2 时，names=['A', 'B']
        names = list("ABC")[:N]
        # 使用 arrays 和 names 创建一个 MultiIndex 对象
        result = MultiIndex.from_arrays(arrays=arrays, names=names)
        # 创建一个预期的 MultiIndex 对象，其中 levels 和 codes 都是包含 N 个空列表的列表，names 是之前定义的 names 列表
        expected = MultiIndex(levels=[[]] * N, codes=[[]] * N, names=names)
        # 使用 tm.assert_index_equal 检查 result 和 expected 是否相等
        tm.assert_index_equal(result, expected)
# 使用 pytest.mark.parametrize 装饰器为 test_from_arrays_invalid_input 函数注入多组参数进行参数化测试
@pytest.mark.parametrize(
    "invalid_sequence_of_arrays",
    [
        1,                # 单个整数
        [1],              # 单个包含一个整数的列表
        [1, 2],           # 包含两个整数的列表
        [[1], 2],         # 第一个元素是列表，第二个元素是整数
        [1, [2]],         # 第一个元素是整数，第二个元素是列表
        "a",              # 单个字符串
        ["a"],            # 单个包含一个字符串的列表
        ["a", "b"],       # 包含两个字符串的列表
        [["a"], "b"],     # 第一个元素是列表，第二个元素是字符串
        (1,),             # 单个包含一个整数的元组
        (1, 2),           # 包含两个整数的元组
        ([1], 2),         # 第一个元素是列表，第二个元素是整数
        (1, [2]),         # 第一个元素是整数，第二个元素是列表
        ("a",),           # 单个包含一个字符串的元组
        ("a", "b"),       # 包含两个字符串的元组
        (["a"], "b"),     # 第一个元素是列表，第二个元素是字符串
        [(1,), 2],        # 第一个元素是元组，第二个元素是整数
        [1, (2,)],        # 第一个元素是整数，第二个元素是元组
        [("a",), "b"],    # 第一个元素是元组，第二个元素是字符串
        ((1,), 2),        # 第一个元素是元组，第二个元素是整数
        (1, (2,)),        # 第一个元素是整数，第二个元素是元组
        (("a",), "b"),    # 第一个元素是元组，第二个元素是字符串
    ],
)
def test_from_arrays_invalid_input(invalid_sequence_of_arrays):
    # 设置错误消息，断言 MultiIndex.from_arrays 方法对于无效输入会抛出 TypeError 异常，并匹配特定错误消息
    msg = "Input must be a list / sequence of array-likes"
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_arrays(arrays=invalid_sequence_of_arrays)


# 使用 pytest.mark.parametrize 装饰器为 test_from_arrays_different_lengths 函数注入多组参数进行参数化测试
@pytest.mark.parametrize(
    "idx1, idx2", [([1, 2, 3], ["a", "b"]), ([], ["a", "b"]), ([1, 2, 3], [])]
)
def test_from_arrays_different_lengths(idx1, idx2):
    # 见 issue gh-13599，设置错误消息，断言 MultiIndex.from_arrays 方法对于长度不同的输入会抛出 ValueError 异常，并匹配特定错误消息
    msg = "^all arrays must be same length$"
    with pytest.raises(ValueError, match=msg):
        MultiIndex.from_arrays([idx1, idx2])


# 测试 MultiIndex.from_arrays 方法能够正确处理 names 参数为 None 的情况
def test_from_arrays_respects_none_names():
    # 见 issue GH27292
    # 创建两个 Series 对象 a 和 b，分别指定它们的名称为 "foo" 和 "bar"
    a = Series([1, 2, 3], name="foo")
    b = Series(["a", "b", "c"], name="bar")

    # 调用 MultiIndex.from_arrays 方法，将 a 和 b 作为数组传入，并指定 names 参数为 None
    result = MultiIndex.from_arrays([a, b], names=None)
    # 创建预期的 MultiIndex 对象 expected，使用 levels 和 codes 分别指定数据和索引
    expected = MultiIndex(
        levels=[[1, 2, 3], ["a", "b", "c"]],
        codes=[[0, 1, 2], [0, 1, 2]],
        names=None
    )

    # 使用 assert_index_equal 方法断言 result 和 expected 相等
    tm.assert_index_equal(result, expected)


# ----------------------------------------------------------------------------
# from_tuples
# ----------------------------------------------------------------------------
# 测试 MultiIndex.from_tuples 方法对于空列表输入会抛出 TypeError 异常，并匹配特定错误消息
def test_from_tuples():
    msg = "Cannot infer number of levels from empty list"
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_tuples([])

    # 创建预期的 MultiIndex 对象 expected，使用 levels 和 codes 分别指定数据和索引
    expected = MultiIndex(
        levels=[[1, 3], [2, 4]],
        codes=[[0, 1], [0, 1]],
        names=["a", "b"]
    )

    # 调用 MultiIndex.from_tuples 方法，将 ((1, 2), (3, 4)) 作为元组传入，并指定 names 参数为 ["a", "b"]
    result = MultiIndex.from_tuples(((1, 2), (3, 4)), names=["a", "b"])
    # 使用 assert_index_equal 方法断言 result 和 expected 相等
    tm.assert_index_equal(result, expected)


# 测试 MultiIndex.from_tuples 方法能够正确处理输入为迭代器的情况
def test_from_tuples_iterator():
    # 见 issue GH 18434
    # 创建预期的 MultiIndex 对象 expected，使用 levels 和 codes 分别指定数据和索引
    expected = MultiIndex(
        levels=[[1, 3], [2, 4]],
        codes=[[0, 1], [0, 1]],
        names=["a", "b"]
    )

    # 调用 MultiIndex.from_tuples 方法，将 zip([1, 3], [2, 4]) 作为迭代器传入，并指定 names 参数为 ["a", "b"]
    result = MultiIndex.from_tuples(zip([1, 3], [2, 4]), names=["a", "b"])
    # 使用 assert_index_equal 方法断言 result 和 expected 相等
    tm.assert_index_equal(result, expected)

    # 设置错误消息，断言 MultiIndex.from_tuples 方法对于非迭代器输入会抛出 TypeError 异常，并匹配特定错误消息
    msg = "Input must be a list / sequence of tuple-likes."
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_tuples(0)


# 测试 MultiIndex.from_tuples 方法对于空列表输入会返回符合预期的 MultiIndex 对象
def test_from_tuples_empty():
    # 见 issue GH 16777
    # 调用 MultiIndex.from_tuples 方法，将空列表作为输入，并指定 names 参数为 ["a", "b"]
    result = MultiIndex.from_tuples([], names=["a", "b"])
    # 调用 MultiIndex.from_arrays 方法，创建预期的 MultiIndex 对象 expected，使用空列表作为数组输入，并指定 names 参数为 ["a", "b"]
    expected = MultiIndex.from_arrays(arrays=[[], []], names=["a", "b"])
    # 使用 assert_index_equal 方法断言 result 和 expected 相等
    tm.assert_index_equal(result, expected)


# 测试 MultiIndex.from_tuples 方法能够正确处理输入为 idx 的情况
def test_from_tuples_index_values(idx):
    # 调用 MultiIndex.from_tuples 方法，将 idx 作为输入，生成 result 对象
    result = MultiIndex.from_tuples(idx)
    # 断言 result 对象的 values 属性与 idx 对象的 values 属性相等
    assert (result.values == idx.values).all()


# 测试 MultiIndex 对象的元组表示，要求其名称必须为列表形式
def test_tuples_with_name_string():
    # 见 issue GH 15110 and GH 14848
    # 创建包含三个元组的列表 li
    li = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    # 设置错误消息，断言 MultiIndex.from_tuples 方法对于名称不是列表形式的情况会抛出 TypeError 异常
    msg = "Names should be list-like for a MultiIndex"
    # 使用 pytest 的上下文管理器检查是否抛出 ValueError 异常，并验证异常消息是否匹配变量 msg
    with pytest.raises(ValueError, match=msg):
        # 在 Index 类中创建实例，传入参数 li 和 name="abc"
        Index(li, name="abc")
    # 使用 pytest 的上下文管理器检查是否抛出 ValueError 异常，并验证异常消息是否匹配变量 msg
    with pytest.raises(ValueError, match=msg):
        # 在 Index 类中创建实例，传入参数 li 和 name="a"
        Index(li, name="a")
def test_from_tuples_with_tuple_label():
    # GH 15457
    # 创建预期的 DataFrame，包含两行数据，带有元组标签
    expected = pd.DataFrame(
        [[2, 1, 2], [4, (1, 2), 3]], columns=["a", "b", "c"]
    ).set_index(["a", "b"])
    # 使用 MultiIndex.from_tuples 创建索引对象 idx
    idx = MultiIndex.from_tuples([(2, 1), (4, (1, 2))], names=("a", "b"))
    # 创建结果 DataFrame，包含一行数据，带有 MultiIndex 索引
    result = pd.DataFrame([2, 3], columns=["c"], index=idx)
    # 使用 assert_frame_equal 断言预期结果和实际结果相等
    tm.assert_frame_equal(expected, result)


# ----------------------------------------------------------------------------
# from_product
# ----------------------------------------------------------------------------
def test_from_product_empty_zero_levels():
    # 0 levels
    # 测试 MultiIndex.from_product 对于空列表的行为，预期引发 ValueError 异常
    msg = "Must pass non-zero number of levels/codes"
    with pytest.raises(ValueError, match=msg):
        MultiIndex.from_product([])


def test_from_product_empty_one_level():
    # 创建空列表的 MultiIndex，单级别
    result = MultiIndex.from_product([[]], names=["A"])
    # 创建预期的 Index 对象，空的，带有名称 "A"
    expected = Index([], name="A")
    # 使用 assert_index_equal 断言结果与预期相等
    tm.assert_index_equal(result.levels[0], expected)
    # 断言结果的名称与预期相等
    assert result.names == ["A"]


@pytest.mark.parametrize(
    "first, second", [([], []), (["foo", "bar", "baz"], []), ([], ["a", "b", "c"])]
)
def test_from_product_empty_two_levels(first, second):
    # 创建空列表的 MultiIndex，双级别
    names = ["A", "B"]
    result = MultiIndex.from_product([first, second], names=names)
    # 创建预期的 MultiIndex，结构与输入列表对应
    expected = MultiIndex(levels=[first, second], codes=[[], []], names=names)
    # 使用 assert_index_equal 断言结果与预期相等
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("N", list(range(4)))
def test_from_product_empty_three_levels(N):
    # GH12258
    # 创建空列表的 MultiIndex，三级别
    names = ["A", "B", "C"]
    lvl2 = list(range(N))
    result = MultiIndex.from_product([[], lvl2, []], names=names)
    # 创建预期的 MultiIndex，结构与输入列表对应
    expected = MultiIndex(levels=[[], lvl2, []], codes=[[], [], []], names=names)
    # 使用 assert_index_equal 断言结果与预期相等
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "invalid_input", [1, [1], [1, 2], [[1], 2], "a", ["a"], ["a", "b"], [["a"], "b"]]
)
def test_from_product_invalid_input(invalid_input):
    # 测试 MultiIndex.from_product 对于不合法输入的行为，预期引发 TypeError 异常
    msg = r"Input must be a list / sequence of iterables|Input must be list-like"
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_product(iterables=invalid_input)


def test_from_product_datetimeindex():
    # 创建日期时间索引的 MultiIndex
    dt_index = date_range("2000-01-01", periods=2)
    mi = MultiIndex.from_product([[1, 2], dt_index])
    # 构建预期的数据结构
    etalon = construct_1d_object_array_from_listlike(
        [
            (1, Timestamp("2000-01-01")),
            (1, Timestamp("2000-01-02")),
            (2, Timestamp("2000-01-01")),
            (2, Timestamp("2000-01-02")),
        ]
    )
    # 使用 assert_numpy_array_equal 断言结果的值与预期相等
    tm.assert_numpy_array_equal(mi.values, etalon)


def test_from_product_rangeindex():
    # RangeIndex 通过 factorize 保留，因此在 levels 中保留
    rng = Index(range(5))
    other = ["a", "b"]
    mi = MultiIndex.from_product([rng, other])
    # 使用 assert_index_equal 断言结果与预期相等
    tm.assert_index_equal(mi._levels[0], rng, exact=True)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("f", [lambda x: x, lambda x: Series(x), lambda x: x.values])
def test_from_product_index_series_categorical(ordered, f):
    # GH13743
    first = ["foo", "bar"]
    # 测试 MultiIndex.from_product 对于 index, series, categorical 类型的行为
    # 创建一个基于类别的 Pandas CategoricalIndex，从字符列表 ["abcaab"] 中生成，指定类别顺序
    idx = pd.CategoricalIndex(list("abcaab"), categories=list("bac"), ordered=ordered)
    
    # 生成一个期望的 Pandas CategoricalIndex，将字符列表 ["abcaab"] 重复一次，并指定类别顺序
    expected = pd.CategoricalIndex(
        list("abcaab") + list("abcaab"), categories=list("bac"), ordered=ordered
    )
    
    # 使用 MultiIndex 的 from_product 方法生成一个索引，其中包含 first 的每个元素与 f(idx) 的乘积
    result = MultiIndex.from_product([first, f(idx)])
    
    # 使用 tm.assert_index_equal 验证 result 的第二个级别的值是否与期望的 expected 相同
    tm.assert_index_equal(result.get_level_values(1), expected)
def test_from_product():
    # 定义第一个列表
    first = ["foo", "bar", "buz"]
    # 定义第二个列表
    second = ["a", "b", "c"]
    # 定义名称列表
    names = ["first", "second"]
    # 使用 from_product 方法创建 MultiIndex 对象，并指定名称
    result = MultiIndex.from_product([first, second], names=names)

    # 预期的元组列表
    tuples = [
        ("foo", "a"),
        ("foo", "b"),
        ("foo", "c"),
        ("bar", "a"),
        ("bar", "b"),
        ("bar", "c"),
        ("buz", "a"),
        ("buz", "b"),
        ("buz", "c"),
    ]
    # 使用 from_tuples 方法创建预期的 MultiIndex 对象，指定名称
    expected = MultiIndex.from_tuples(tuples, names=names)

    # 断言两个 MultiIndex 对象是否相等
    tm.assert_index_equal(result, expected)


def test_from_product_iterator():
    # GH 18434
    # 定义第一个列表
    first = ["foo", "bar", "buz"]
    # 定义第二个列表
    second = ["a", "b", "c"]
    # 定义名称列表
    names = ["first", "second"]
    # 预期的元组列表
    tuples = [
        ("foo", "a"),
        ("foo", "b"),
        ("foo", "c"),
        ("bar", "a"),
        ("bar", "b"),
        ("bar", "c"),
        ("buz", "a"),
        ("buz", "b"),
        ("buz", "c"),
    ]
    # 使用 from_tuples 方法创建预期的 MultiIndex 对象，指定名称
    expected = MultiIndex.from_tuples(tuples, names=names)

    # 使用迭代器作为输入创建 MultiIndex 对象，指定名称
    result = MultiIndex.from_product(iter([first, second]), names=names)
    # 断言两个 MultiIndex 对象是否相等
    tm.assert_index_equal(result, expected)

    # 测试无效的非可迭代输入
    msg = "Input must be a list / sequence of iterables."
    # 使用 pytest 的断言检查是否会抛出 TypeError 异常，并匹配错误信息
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_product(0)


@pytest.mark.parametrize(
    "a, b, expected_names",
    [
        (
            Series([1, 2, 3], name="foo"),
            Series(["a", "b"], name="bar"),
            ["foo", "bar"],
        ),
        (Series([1, 2, 3], name="foo"), ["a", "b"], ["foo", None]),
        ([1, 2, 3], ["a", "b"], None),
    ],
)
def test_from_product_infer_names(a, b, expected_names):
    # GH27292
    # 使用 from_product 方法创建 MultiIndex 对象，推断名称
    result = MultiIndex.from_product([a, b])
    # 创建预期的 MultiIndex 对象，指定预期的 levels, codes 和 names
    expected = MultiIndex(
        levels=[[1, 2, 3], ["a", "b"]],
        codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        names=expected_names,
    )
    # 断言两个 MultiIndex 对象是否相等
    tm.assert_index_equal(result, expected)


def test_from_product_respects_none_names():
    # GH27292
    # 创建 Series 对象 a 和 b
    a = Series([1, 2, 3], name="foo")
    b = Series(["a", "b"], name="bar")

    # 使用 from_product 方法创建 MultiIndex 对象，名称设为 None
    result = MultiIndex.from_product([a, b], names=None)
    # 创建预期的 MultiIndex 对象，指定 levels 和 codes，名称为 None
    expected = MultiIndex(
        levels=[[1, 2, 3], ["a", "b"]],
        codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        names=None,
    )
    # 断言两个 MultiIndex 对象是否相等
    tm.assert_index_equal(result, expected)


def test_from_product_readonly():
    # GH#15286 passing read-only array to from_product
    # 创建 NumPy 数组 a 和列表 b
    a = np.array(range(3))
    b = ["a", "b"]
    # 使用 from_product 方法创建 MultiIndex 对象
    expected = MultiIndex.from_product([a, b])

    # 将数组 a 设置为只读
    a.setflags(write=False)
    # 使用 from_product 方法创建 MultiIndex 对象
    result = MultiIndex.from_product([a, b])
    # 断言两个 MultiIndex 对象是否相等
    tm.assert_index_equal(result, expected)


def test_create_index_existing_name(idx):
    # GH11193, when an existing index is passed, and a new name is not
    # specified, the new index should inherit the previous object name
    # 将传入的索引对象赋值给 index
    index = idx
    # 修改索引的名称为 ["foo", "bar"]
    index.names = ["foo", "bar"]
    # 使用 Index 构造函数创建新的 Index 对象
    result = Index(index)
    # 创建预期的索引对象 `expected`，包含元组形式的数据和指定的数据类型
    expected = Index(
        Index(
            [
                ("foo", "one"),
                ("foo", "two"),
                ("bar", "one"),
                ("baz", "two"),
                ("qux", "one"),
                ("qux", "two"),
            ],
            dtype="object",
        )
    )
    # 使用断言函数检查 `result` 和 `expected` 是否相等
    tm.assert_index_equal(result, expected)

    # 创建带有名称 "A" 的索引对象 `result`
    result = Index(index, name="A")
    # 创建预期的索引对象 `expected`，包含元组形式的数据、指定的数据类型以及指定的名称 "A"
    expected = Index(
        Index(
            [
                ("foo", "one"),
                ("foo", "two"),
                ("bar", "one"),
                ("baz", "two"),
                ("qux", "one"),
                ("qux", "two"),
            ],
            dtype="object",
        ),
        name="A",
    )
    # 使用断言函数检查 `result` 和 `expected` 是否相等
    tm.assert_index_equal(result, expected)
# ----------------------------------------------------------------------------
# from_frame
# ----------------------------------------------------------------------------

# 测试 MultiIndex 类的 from_frame 方法，用于从 DataFrame 创建 MultiIndex 对象
def test_from_frame():
    # 创建一个包含两列的 DataFrame
    df = pd.DataFrame(
        [["a", "a"], ["a", "b"], ["b", "a"], ["b", "b"]],
        columns=["L1", "L2"]
    )
    # 创建预期的 MultiIndex 对象
    expected = pd.MultiIndex.from_tuples(
        [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")],
        names=["L1", "L2"]
    )
    # 调用 from_frame 方法生成结果 MultiIndex 对象
    result = pd.MultiIndex.from_frame(df)
    # 断言结果与预期相等
    tm.assert_index_equal(expected, result)


# 测试处理 DataFrame 中存在缺失值的情况下的 from_frame 方法
def test_from_frame_missing_values_multiIndex():
    # 导入 pytest，如果未安装则跳过测试
    pa = pytest.importorskip("pyarrow")

    # 创建包含不同数据类型的 DataFrame，其中包含 Int64 和 Float64 类型的数据
    df = pd.DataFrame(
        {
            "a": pd.Series([1, 2, None], dtype="Int64"),
            "b": pd.Float64Dtype().__from_arrow__(pa.array([0.2, np.nan, None])),
        }
    )
    # 使用 from_frame 方法创建 MultiIndex 对象
    multi_indexed = pd.MultiIndex.from_frame(df)
    # 创建预期的 MultiIndex 对象
    expected = pd.MultiIndex.from_arrays(
        [
            pd.Series([1, 2, None]).astype("Int64"),
            pd.Float64Dtype().__from_arrow__(pa.array([0.2, np.nan, None])),
        ],
        names=["a", "b"],
    )
    # 断言生成的 MultiIndex 对象与预期相等
    tm.assert_index_equal(multi_indexed, expected)


# 使用 pytest 的参数化装饰器，测试处理错误输入时的 from_frame 方法
@pytest.mark.parametrize(
    "non_frame",
    [
        pd.Series([1, 2, 3, 4]),
        [1, 2, 3, 4],
        [[1, 2], [3, 4], [5, 6]],
        pd.Index([1, 2, 3, 4]),
        np.array([[1, 2], [3, 4], [5, 6]]),
        27,
    ],
)
def test_from_frame_error(non_frame):
    # 检查当输入非 DataFrame 时，from_frame 方法是否引发 TypeError 异常
    with pytest.raises(TypeError, match="Input must be a DataFrame"):
        pd.MultiIndex.from_frame(non_frame)


# 测试 from_frame 方法在保持数据类型一致性方面的行为
def test_from_frame_dtype_fidelity():
    # 创建一个包含不同数据类型的 DataFrame
    df = pd.DataFrame(
        {
            "dates": pd.date_range("19910905", periods=6, tz="US/Eastern"),
            "a": [1, 1, 1, 2, 2, 2],
            "b": pd.Categorical(["a", "a", "b", "b", "c", "c"], ordered=True),
            "c": ["x", "x", "y", "z", "x", "y"],
        }
    )
    # 记录原始 DataFrame 的数据类型
    original_dtypes = df.dtypes.to_dict()

    # 使用 from_frame 方法创建 MultiIndex 对象
    expected_mi = pd.MultiIndex.from_arrays(
        [
            pd.date_range("19910905", periods=6, tz="US/Eastern"),
            [1, 1, 1, 2, 2, 2],
            pd.Categorical(["a", "a", "b", "b", "c", "c"], ordered=True),
            ["x", "x", "y", "z", "x", "y"],
        ],
        names=["dates", "a", "b", "c"],
    )
    mi = pd.MultiIndex.from_frame(df)
    # 检查生成的 MultiIndex 对象与预期是否相等
    tm.assert_index_equal(expected_mi, mi)
    
    # 检查生成的 MultiIndex 对象各级别数据类型是否与原始 DataFrame 保持一致
    mi_dtypes = {name: mi.levels[i].dtype for i, name in enumerate(mi.names)}
    assert original_dtypes == mi_dtypes


# 使用 pytest 的参数化装饰器，测试 from_frame 方法在不同有效命名下的行为
@pytest.mark.parametrize(
    "names_in,names_out", [(None, [("L1", "x"), ("L2", "y")]), (["x", "y"], ["x", "y"])]
)
def test_from_frame_valid_names(names_in, names_out):
    # 创建一个包含命名 MultiIndex 的 DataFrame
    df = pd.DataFrame(
        [["a", "a"], ["a", "b"], ["b", "a"], ["b", "b"]],
        columns=pd.MultiIndex.from_tuples([("L1", "x"), ("L2", "y")]),
    )
    # 使用 from_frame 方法创建 MultiIndex 对象，并指定命名
    mi = pd.MultiIndex.from_frame(df, names=names_in)
    # 检查生成的 MultiIndex 对象的命名是否与预期相符
    assert mi.names == names_out
    [
        # 错误示例，第一个元素应该是列表形式以创建 MultiIndex
        ("bad_input", "Names should be list-like for a MultiIndex"),
        # 正确示例，这里的列表包含了 MultiIndex 的每个级别的名称
        (["a", "b", "c"], "Length of names must match number of levels in MultiIndex"),
    ],
# 定义一个测试函数，用于测试 MultiIndex.from_frame 方法对无效名称的处理
def test_from_frame_invalid_names(names, expected_error_msg):
    # 创建一个 DataFrame，包含四行两列的数据，列名为 ("L1", "x") 和 ("L2", "y")
    df = pd.DataFrame(
        [["a", "a"], ["a", "b"], ["b", "a"], ["b", "b"]],
        columns=MultiIndex.from_tuples([("L1", "x"), ("L2", "y")]),
    )
    # 使用 pytest 断言，验证调用 MultiIndex.from_frame 方法时是否会抛出 ValueError 异常，并匹配预期的错误信息
    with pytest.raises(ValueError, match=expected_error_msg):
        MultiIndex.from_frame(df, names=names)


def test_index_equal_empty_iterable():
    # #16844
    # 创建一个空的 MultiIndex 对象 a，使用空的 levels 和 codes，指定名称为 ["a", "b"]
    a = MultiIndex(levels=[[], []], codes=[[], []], names=["a", "b"])
    # 从空数组创建一个 MultiIndex 对象 b，指定名称为 ["a", "b"]
    b = MultiIndex.from_arrays(arrays=[[], []], names=["a", "b"])
    # 使用 tm.assert_index_equal 断言 a 和 b 对象相等
    tm.assert_index_equal(a, b)


def test_raise_invalid_sortorder():
    # Test that the MultiIndex constructor raise when a incorrect sortorder is given
    # GH#28518

    # 定义一个 levels 列表，包含两个子列表
    levels = [[0, 1], [0, 1, 2]]

    # 使用正确的 sortorder 创建一个 MultiIndex 对象
    MultiIndex(
        levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]], sortorder=2
    )

    # 使用 pytest 断言，验证当给定不正确的 sortorder 时，MultiIndex 构造函数是否会抛出 ValueError 异常，并匹配特定的错误信息
    with pytest.raises(ValueError, match=r".* sortorder 2 with lexsort_depth 1.*"):
        MultiIndex(
            levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 2, 1]], sortorder=2
        )

    with pytest.raises(ValueError, match=r".* sortorder 1 with lexsort_depth 0.*"):
        MultiIndex(
            levels=levels, codes=[[0, 0, 1, 0, 1, 1], [0, 1, 0, 2, 2, 1]], sortorder=1
        )


def test_datetimeindex():
    # 创建一个带时区信息的 DatetimeIndex 对象 idx1
    idx1 = pd.DatetimeIndex(
        ["2013-04-01 9:00", "2013-04-02 9:00", "2013-04-03 9:00"] * 2, tz="Asia/Tokyo"
    )
    # 创建一个时区为 US/Eastern 的 DatetimeIndex 对象 idx2
    idx2 = date_range("2010/01/01", periods=6, freq="ME", tz="US/Eastern")
    # 使用 idx1 和 idx2 创建一个 MultiIndex 对象 idx
    idx = MultiIndex.from_arrays([idx1, idx2])

    # 创建一个预期的 DatetimeIndex 对象 expected1，指定时区为 Asia/Tokyo
    expected1 = pd.DatetimeIndex(
        ["2013-04-01 9:00", "2013-04-02 9:00", "2013-04-03 9:00"], tz="Asia/Tokyo"
    )

    # 使用 tm.assert_index_equal 断言 idx.levels[0] 和 expected1 相等
    tm.assert_index_equal(idx.levels[0], expected1)
    # 使用 tm.assert_index_equal 断言 idx.levels[1] 和 idx2 相等

    # from datetime combos
    # GH 7888
    # 创建三个不同的日期时间对象 date1, date2, date3
    date1 = np.datetime64("today")
    date2 = datetime.today()
    date3 = Timestamp.today()

    # 使用 itertools.product 对三个日期时间对象进行排列组合，并创建 MultiIndex 对象 index
    for d1, d2 in itertools.product([date1, date2, date3], [date1, date2, date3]):
        index = MultiIndex.from_product([[d1], [d2]])
        # 使用断言验证 index.levels[0] 和 index.levels[1] 是 pd.DatetimeIndex 类型的对象

    # 但是 date 对象不会创建 DatetimeIndex 对象，与 Index 的行为相匹配
    # 创建一个包含 date4 和 date2 的 MultiIndex 对象 index
    assert not isinstance(index.levels[0], pd.DatetimeIndex)
    assert isinstance(index.levels[1], pd.DatetimeIndex)


def test_constructor_with_tz():
    # 创建一个带有时区信息的 DatetimeIndex 对象 index
    index = pd.DatetimeIndex(
        ["2013/01/01 09:00", "2013/01/02 09:00"], name="dt1", tz="US/Pacific"
    )
    # 创建一个带有时区信息的 DatetimeIndex 对象 columns
    columns = pd.DatetimeIndex(
        ["2014/01/01 09:00", "2014/01/02 09:00"], name="dt2", tz="Asia/Tokyo"
    )

    # 使用 MultiIndex.from_arrays 创建一个 MultiIndex 对象 result，包含 index 和 columns
    result = MultiIndex.from_arrays([index, columns])

    # 使用断言验证 result 的名称为 ["dt1", "dt2"]
    assert result.names == ["dt1", "dt2"]
    # 使用 tm.assert_index_equal 断言 result.levels[0] 和 index 相等
    tm.assert_index_equal(result.levels[0], index)
    # 使用 tm.assert_index_equal 断言 result.levels[1] 和 columns 相等

    # 使用 MultiIndex.from_arrays 创建一个 MultiIndex 对象 result，包含 index 和 columns 的 Series 版本
    result = MultiIndex.from_arrays([Series(index), Series(columns)])

    # 使用断言验证 result 的名称为 ["dt1", "dt2"]
    assert result.names == ["dt1", "dt2"]
    # 使用测试工具tm断言result.levels[0]与index的索引是否相等
    tm.assert_index_equal(result.levels[0], index)
    # 使用测试工具tm断言result.levels[1]与columns的索引是否相等
    tm.assert_index_equal(result.levels[1], columns)
def test_multiindex_inference_consistency():
    # 检查推断行为是否与基类匹配

    v = date.today()  # 获取当天日期对象

    arr = [v, v]  # 创建包含两个日期对象的列表

    idx = Index(arr)  # 使用日期对象列表创建索引对象
    assert idx.dtype == object  # 断言索引对象的数据类型为 object 类型

    mi = MultiIndex.from_arrays([arr])  # 使用日期对象列表创建多重索引对象
    lev = mi.levels[0]  # 获取多重索引对象的第一个级别
    assert lev.dtype == object  # 断言第一个级别的数据类型为 object 类型

    mi = MultiIndex.from_product([arr])  # 使用日期对象列表创建笛卡尔积多重索引对象
    lev = mi.levels[0]  # 获取多重索引对象的第一个级别
    assert lev.dtype == object  # 断言第一个级别的数据类型为 object 类型

    mi = MultiIndex.from_tuples([(x,) for x in arr])  # 使用日期对象列表创建元组形式的多重索引对象
    lev = mi.levels[0]  # 获取多重索引对象的第一个级别
    assert lev.dtype == object  # 断言第一个级别的数据类型为 object 类型


def test_dtype_representation(using_infer_string):
    # GH#46900

    pmidx = MultiIndex.from_arrays([[1], ["a"]], names=[("a", "b"), ("c", "d")])
    # 使用指定数组创建具有名称的多重索引对象

    result = pmidx.dtypes  # 获取多重索引对象的数据类型

    exp = "object" if not using_infer_string else "string"
    # 如果未使用推断字符串，则期望为 "object"，否则为 "string"

    expected = Series(
        ["int64", exp],
        index=MultiIndex.from_tuples([("a", "b"), ("c", "d")]),
        dtype=object,
    )
    # 创建期望的 Series 对象，包含特定的数据类型和索引

    tm.assert_series_equal(result, expected)
    # 使用测试工具断言结果 Series 与期望的 Series 相等
```