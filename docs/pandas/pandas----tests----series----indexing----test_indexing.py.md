# `D:\src\scipysrc\pandas\pandas\tests\series\indexing\test_indexing.py`

```
"""test get/set & misc"""

# 导入必要的库
from datetime import timedelta
import re

import numpy as np
import pytest

# 导入 pandas 库的特定错误处理类
from pandas.errors import IndexingError

# 导入 pandas 库中的特定模块和类
from pandas import (
    NA,
    DataFrame,
    Index,
    IndexSlice,
    MultiIndex,
    NaT,
    Series,
    Timedelta,
    Timestamp,
    concat,
    date_range,
    isna,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


# 定义测试函数 test_basic_indexing
def test_basic_indexing():
    # 创建一个 Series 对象，包含随机数，并指定索引
    s = Series(
        np.random.default_rng(2).standard_normal(5), index=["a", "b", "a", "a", "b"]
    )

    # 测试索引不存在的情况是否引发 KeyError 异常
    with pytest.raises(KeyError, match="^5$"):
        s[5]

    # 测试索引值为 'c' 的情况是否引发 KeyError 异常
    with pytest.raises(KeyError, match=r"^'c'$"):
        s["c"]

    # 对 Series 对象按索引排序
    s = s.sort_index()

    # 再次测试索引不存在的情况是否引发 KeyError 异常
    with pytest.raises(KeyError, match="^5$"):
        s[5]


# 定义测试函数 test_getitem_numeric_should_not_fallback_to_positional
def test_getitem_numeric_should_not_fallback_to_positional(any_numeric_dtype):
    # 设置变量 dtype 为任意数值类型
    dtype = any_numeric_dtype
    # 创建一个具有指定索引和数值的 Series 对象
    idx = Index([1, 0, 1], dtype=dtype)
    ser = Series(range(3), index=idx)
    # 对索引为 1 的元素进行获取操作
    result = ser[1]
    # 期望结果是另一个 Series 对象
    expected = Series([0, 2], index=Index([1, 1], dtype=dtype))
    tm.assert_series_equal(result, expected, check_exact=True)


# 定义测试函数 test_setitem_numeric_should_not_fallback_to_positional
def test_setitem_numeric_should_not_fallback_to_positional(any_numeric_dtype):
    # 设置变量 dtype 为任意数值类型
    dtype = any_numeric_dtype
    # 创建一个具有指定索引和数值的 Series 对象
    idx = Index([1, 0, 1], dtype=dtype)
    ser = Series(range(3), index=idx)
    # 对索引为 1 的元素进行赋值操作
    ser[1] = 10
    # 期望结果是另一个 Series 对象
    expected = Series([10, 1, 10], index=idx)
    tm.assert_series_equal(ser, expected, check_exact=True)


# 定义测试函数 test_basic_getitem_with_labels
def test_basic_getitem_with_labels(datetime_series):
    # 获取时间序列的特定索引
    indices = datetime_series.index[[5, 10, 15]]

    # 通过特定索引获取 Series 的子集
    result = datetime_series[indices]
    expected = datetime_series.reindex(indices)
    tm.assert_series_equal(result, expected)

    # 通过切片方式获取 Series 的子集
    result = datetime_series[indices[0] : indices[2]]
    expected = datetime_series.loc[indices[0] : indices[2]]
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_basic_getitem_dt64tz_values
def test_basic_getitem_dt64tz_values():
    # 创建具有时区信息的 Series 对象
    ser = Series(
        date_range("2011-01-01", periods=3, tz="US/Eastern"), index=["a", "b", "c"]
    )
    expected = Timestamp("2011-01-01", tz="US/Eastern")
    # 测试通过不同方式获取 Series 中的元素
    result = ser.loc["a"]
    assert result == expected
    result = ser.iloc[0]
    assert result == expected
    result = ser["a"]
    assert result == expected


# 定义测试函数 test_getitem_setitem_ellipsis
def test_getitem_setitem_ellipsis():
    # 创建一个包含随机数的 Series 对象
    s = Series(np.random.default_rng(2).standard_normal(10))

    # 测试通过省略符获取整个 Series 的内容
    result = s[...]
    tm.assert_series_equal(result, s)


# 使用 pytest 的参数化装饰器执行多组参数的测试
@pytest.mark.parametrize(
    "result_1, duplicate_item, expected_1",
    [
        [
            {1: 12, 2: [1, 2, 2, 3]},
            {1: 313},
            Series({1: 12}, dtype=object),
        ],
        [
            {1: [1, 2, 3], 2: [1, 2, 2, 3]},
            {1: [1, 2, 3]},
            Series({1: [1, 2, 3]}),
        ],
    ],
)
# 定义测试函数 test_getitem_with_duplicates_indices
def test_getitem_with_duplicates_indices(result_1, duplicate_item, expected_1):
    # 创建两个 Series 对象
    result_1 = Series(result_1)
    duplicate_item = Series(duplicate_item)
    # 对其中一个 Series 进行拼接操作
    result = result_1._append(duplicate_item)
    expected = expected_1._append(duplicate_item)
    # 使用测试工具对结果的第二个元素进行断言，验证其与期望值是否相等
        tm.assert_series_equal(result[1], expected)
    # 断言结果的第三个元素是否等于result_1的第三个元素，用于简单的值比较
        assert result[2] == result_1[2]
# 测试函数：测试通过整数索引和标签索引访问和设置 Series 对象的功能
def test_getitem_setitem_integers():
    # 创建一个包含整数和对应标签的 Series 对象
    s = Series([1, 2, 3], ["a", "b", "c"])

    # 使用 iloc 通过整数索引访问和比较值
    assert s.iloc[0] == s["a"]
    
    # 使用 iloc 通过整数索引设置值
    s.iloc[0] = 5
    
    # 使用标签索引确认设置成功
    tm.assert_almost_equal(s["a"], 5)


# 测试函数：测试 Series 对象在不同索引方式下存储时间戳对象的行为
def test_series_box_timestamp():
    # 创建一个日期范围对象
    rng = date_range("20090415", "20090519", freq="B")
    
    # 创建一个 Series 对象，使用日期范围作为数据
    ser = Series(rng)
    
    # 检查 Series 中第一个元素是否为 Timestamp 类型
    assert isinstance(ser[0], Timestamp)
    
    # 使用 at 方法通过标签访问元素，确认返回类型为 Timestamp
    assert isinstance(ser.at[1], Timestamp)
    
    # 使用 iat 方法通过整数位置访问元素，确认返回类型为 Timestamp
    assert isinstance(ser.iat[2], Timestamp)
    
    # 使用 loc 方法通过标签访问元素，确认返回类型为 Timestamp
    assert isinstance(ser.loc[3], Timestamp)
    
    # 使用 iloc 方法通过整数位置访问元素，确认返回类型为 Timestamp
    assert isinstance(ser.iloc[4], Timestamp)

    # 创建一个索引也为日期范围的 Series 对象，确认不同索引方式下的行为一致
    ser = Series(rng, index=rng)
    
    # 使用日期范围对象作为标签访问元素，确认返回类型为 Timestamp
    assert isinstance(ser[rng[0]], Timestamp)
    
    # 使用 at 方法通过标签访问元素，确认返回类型为 Timestamp
    assert isinstance(ser.at[rng[1]], Timestamp)
    
    # 使用 iat 方法通过整数位置访问元素，确认返回类型为 Timestamp
    assert isinstance(ser.iat[2], Timestamp)
    
    # 使用 loc 方法通过标签访问元素，确认返回类型为 Timestamp
    assert isinstance(ser.loc[rng[3]], Timestamp)
    
    # 使用 iloc 方法通过整数位置访问元素，确认返回类型为 Timestamp
    assert isinstance(ser.iloc[4], Timestamp)


# 测试函数：测试 Series 对象在不同索引方式下存储时间差对象的行为
def test_series_box_timedelta():
    # 创建一个时间差范围对象
    rng = timedelta_range("1 day 1 s", periods=5, freq="h")
    
    # 创建一个 Series 对象，使用时间差范围作为数据
    ser = Series(rng)
    
    # 检查 Series 中第一个元素是否为 Timedelta 类型
    assert isinstance(ser[0], Timedelta)
    
    # 使用 at 方法通过标签访问元素，确认返回类型为 Timedelta
    assert isinstance(ser.at[1], Timedelta)
    
    # 使用 iat 方法通过整数位置访问元素，确认返回类型为 Timedelta
    assert isinstance(ser.iat[2], Timedelta)
    
    # 使用 loc 方法通过标签访问元素，确认返回类型为 Timedelta
    assert isinstance(ser.loc[3], Timedelta)
    
    # 使用 iloc 方法通过整数位置访问元素，确认返回类型为 Timedelta
    assert isinstance(ser.iloc[4], Timedelta)


# 测试函数：测试 Series 对象通过不明确的键引发 KeyError 的行为
def test_getitem_ambiguous_keyerror(indexer_sl):
    # 创建一个索引为偶数的 Series 对象
    ser = Series(range(10), index=list(range(0, 20, 2)))
    
    # 使用不在索引中的键访问元素，确认引发 KeyError 异常
    with pytest.raises(KeyError, match=r"^1$"):
        indexer_sl(ser)[1]


# 测试函数：测试 Series 对象包含重复值和缺失索引的行为
def test_getitem_dups_with_missing(indexer_sl):
    # 创建一个包含重复索引的 Series 对象
    ser = Series([1, 2, 3, 4], ["foo", "bar", "foo", "bah"])
    
    # 使用不在索引中的多个键访问元素，确认引发 KeyError 异常，并检查异常信息
    with pytest.raises(KeyError, match=re.escape("['bam'] not in index")):
        indexer_sl(ser)[["foo", "bar", "bah", "bam"]]


# 测试函数：测试 Series 对象在包含重复索引并设置不明确键时的行为
def test_setitem_ambiguous_keyerror(indexer_sl):
    # 创建一个索引为偶数的 Series 对象
    s = Series(range(10), index=list(range(0, 20, 2)))
    
    # 创建副本对象以进行设置操作
    s2 = s.copy()
    
    # 使用不在索引中的键设置元素，相当于追加操作
    indexer_sl(s2)[1] = 5
    
    # 创建预期结果对象，包含追加的元素
    expected = concat([s, Series([5], index=[1])])
    
    # 检查设置后的 Series 对象与预期结果对象是否相等
    tm.assert_series_equal(s2, expected)


# 测试函数：测试 Series 对象的元素设置行为
def test_setitem(datetime_series):
    # 使用标签访问元素并设置为 NaN
    datetime_series[datetime_series.index[5]] = np.nan
    
    # 使用整数位置列表访问元素并设置为 NaN
    datetime_series.iloc[[1, 2, 17]] = np.nan
    
    # 使用整数位置访问元素并设置为 NaN，然后检查是否为 NaN
    datetime_series.iloc[6] = np.nan
    assert np.isnan(datetime_series.iloc[6])
    
    # 检查元素是否设置成功
    assert np.isnan(datetime_series.iloc[2])
    
    # 使用布尔索引访问元素并设置非 NaN 值
    datetime_series[np.isnan(datetime_series)] = 5
    
    # 检查元素是否设置成功
    assert not np.isnan(datetime_series.iloc[2])


# 测试函数：测试 Series 对象切片行为
def test_setslice(datetime_series):
    # 对 Series 对象进行切片并检查长度与索引是否一致
    sl = datetime_series[5:20]
    assert len(sl) == len(sl.index)
    
    # 检查切片后的索引是否唯一
    assert sl.index.is_unique is True


# 测试函数：测试 Series 对象不同方式的索引行为
def test_basic_getitem_setitem_corner(datetime_series):
    # 尝试使用无效的元组作为键引发 KeyError 异常
    msg = "key of type tuple not found and not a MultiIndex"
    with pytest.raises(KeyError, match=msg):
        datetime_series[:, 2]
    with pytest.raises(KeyError, match=msg):
        datetime_series[:, 2] = 2

    # 尝试使用奇怪的列表作为索引引发 ValueError 异常
    msg = "Indexing with a single-item list"
    with pytest.raises(ValueError, match=msg):
        # GH#31299
        datetime_series[[slice(None, 5)]]
    # 从 datetime_series 中获取切片 (slice(None, 5),) 并赋值给 result
    result = datetime_series[(slice(None, 5),)]
    # 从 datetime_series 中获取切片 [:5] 并赋值给 expected
    expected = datetime_series[:5]
    # 使用 pytest 的 assert_series_equal 函数比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)

    # 设置预期的错误消息字符串，检查是否会引发 TypeError 并匹配消息
    msg = r"unhashable type(: 'slice')?"
    # 使用 pytest 的 raises 函数检查是否会抛出 TypeError，并且错误消息匹配 msg
    with pytest.raises(TypeError, match=msg):
        # 尝试使用非可哈希的类型作为索引来访问 datetime_series
        datetime_series[[5, [None, None]]]
    # 同样检查赋值时的情况
    with pytest.raises(TypeError, match=msg):
        # 尝试使用非可哈希的类型作为索引来赋值给 datetime_series
        datetime_series[[5, [None, None]]] = 2
# 复制输入的 string_series，以便稍后比较
original = string_series.copy()

# 从 string_series 中获取索引位置为 10 到 19 的子序列
numSlice = string_series[10:20]

# 从 string_series 中获取最后 10 个元素的子序列
numSliceEnd = string_series[-10:]

# 从 object_series 中获取索引位置为 10 到 19 的子序列
objSlice = object_series[10:20]

# 断言：string_series 的第 9 个索引不在 numSlice 的索引中
assert string_series.index[9] not in numSlice.index

# 断言：object_series 的第 9 个索引不在 objSlice 的索引中
assert object_series.index[9] not in objSlice.index

# 断言：numSlice 的长度等于其索引的长度
assert len(numSlice) == len(numSlice.index)

# 断言：string_series 中 numSlice 的第一个索引位置对应的值等于 numSlice 中相同索引位置的值
assert string_series[numSlice.index[0]] == numSlice[numSlice.index[0]]

# 断言：numSlice 的第二个索引位置应该等于 string_series 的索引位置 11
assert numSlice.index[1] == string_series.index[11]

# 使用 numpy 断言：np.array(numSliceEnd) 应与 np.array(string_series)[-10:] 相等
tm.assert_numpy_array_equal(np.array(numSliceEnd), np.array(string_series)[-10:])

# 测试返回视图
sl = string_series[10:20]
sl[:] = 0

# 断言：修改 sl 不会修改原始的 string_series，因为采用了写时复制（CoW）的策略
tm.assert_series_equal(string_series, original)
    # GH 35534 - 当 Series 的索引是元组时，选择其对应的数值
    # 创建一个 Series 对象，指定元素为 [1, 2]，索引为 [("a",), ("b",)]
    s = Series([1, 2], index=[("a",), ("b",)])
    
    # 断言，验证选择索引为 ("a",) 的值为 1
    assert s[("a",)] == 1
    
    # 断言，验证选择索引为 ("b",) 的值为 2
    assert s[("b",)] == 2
    
    # 修改索引为 ("b",) 的值为 3
    s[("b",)] = 3
    
    # 断言，验证修改后选择索引为 ("b",) 的值为 3
    assert s[("b",)] == 3
def test_frozenset_index():
    # GH35747 - Selecting values when a Series has an Index of frozenset
    # 创建两个 frozenset 作为索引
    idx0, idx1 = frozenset("a"), frozenset("b")
    # 创建一个 Series 对象，指定索引为 idx0 和 idx1，对应数值为 1 和 2
    s = Series([1, 2], index=[idx0, idx1])
    # 断言索引为 idx0 的值为 1
    assert s[idx0] == 1
    # 断言索引为 idx1 的值为 2
    assert s[idx1] == 2
    # 修改索引为 idx1 的值为 3
    s[idx1] = 3
    # 再次断言索引为 idx1 的值为 3
    assert s[idx1] == 3


def test_loc_setitem_all_false_indexer():
    # GH#45778
    # 创建一个包含 [1, 2] 的 Series 对象，索引为 ["a", "b"]
    ser = Series([1, 2], index=["a", "b"])
    # 复制 ser 作为预期结果
    expected = ser.copy()
    # 创建另一个 Series 对象 rhs，索引为 ["a", "b"]，值为 [6, 7]
    rhs = Series([6, 7], index=["a", "b"])
    # 使用布尔条件 ser > 100 对 ser 进行索引，将 rhs 赋值给选中的位置
    ser.loc[ser > 100] = rhs
    # 断言修改后的 ser 与预期结果 expected 相等
    tm.assert_series_equal(ser, expected)


def test_loc_boolean_indexer_non_matching_index():
    # GH#46551
    # 创建一个只含有一个元素 1 的 Series 对象
    ser = Series([1])
    # 使用布尔 Series [NA, False] 对 ser 进行索引
    result = ser.loc[Series([NA, False], dtype="boolean")]
    # 创建一个空的预期 Series 对象
    expected = Series([], dtype="int64")
    # 断言结果 result 与预期结果 expected 相等
    tm.assert_series_equal(result, expected)


def test_loc_boolean_indexer_miss_matching_index():
    # GH#46551
    # 创建一个只含有一个元素 1 的 Series 对象
    ser = Series([1])
    # 创建一个布尔 Series [NA, False]，指定索引为 [1, 2]
    indexer = Series([NA, False], dtype="boolean", index=[1, 2])
    # 使用不匹配的索引 indexer 对 ser 进行 loc 操作，预期抛出 IndexingError 异常
    with pytest.raises(IndexingError, match="Unalignable"):
        ser.loc[indexer]


def test_loc_setitem_nested_data_enlargement():
    # GH#48614
    # 创建一个 DataFrame 对象 df，包含一个列 "a"，值为 [1]
    df = DataFrame({"a": [1]})
    # 创建一个 Series 对象 ser，包含一个键 "label"，值为 df
    ser = Series({"label": df})
    # 使用 loc 将 df 添加到 ser 的新键 "new_label"
    ser.loc["new_label"] = df
    # 创建一个预期结果对象，包含键 "label" 和 "new_label"，值均为 df
    expected = Series({"label": df, "new_label": df})
    # 断言修改后的 ser 与预期结果 expected 相等
    tm.assert_series_equal(ser, expected)


def test_loc_ea_numeric_index_oob_slice_end():
    # GH#50161
    # 创建一个 Series 对象 ser，所有元素均为 1，索引为 [0, 1, 2]
    ser = Series(1, index=Index([0, 1, 2], dtype="Int64"))
    # 使用 loc 对索引范围为 2 到 3 的切片，获取结果
    result = ser.loc[2:3]
    # 创建一个预期结果对象，包含索引为 2，值为 1
    expected = Series(1, index=Index([2], dtype="Int64"))
    # 断言结果 result 与预期结果 expected 相等
    tm.assert_series_equal(result, expected)


def test_getitem_bool_int_key():
    # GH#48653
    # 创建一个 Series 对象 ser，键为 True 和 False，值分别为 1 和 0
    ser = Series({True: 1, False: 0})
    # 使用 loc 获取键为 0 的值，预期抛出 KeyError 异常
    with pytest.raises(KeyError, match="0"):
        ser.loc[0]


@pytest.mark.parametrize("val", [{}, {"b": "x"}])
@pytest.mark.parametrize("indexer", [[], [False, False], slice(0, -1), np.array([])])
def test_setitem_empty_indexer(indexer, val):
    # GH#45981
    # 创建一个 DataFrame 对象 df，包含一列 "a"，以及额外的键值对 val
    df = DataFrame({"a": [1, 2], **val})
    # 复制 df 作为预期结果 expected
    expected = df.copy()
    # 使用空的索引 indexer 对 df 进行 loc 操作，将所有选中位置赋值为 1.5
    df.loc[indexer] = 1.5
    # 断言修改后的 df 与预期结果 expected 相等
    tm.assert_frame_equal(df, expected)


class TestDeprecatedIndexers:
    @pytest.mark.parametrize("key", [{1}, {1: 1}])
    def test_getitem_dict_and_set_deprecated(self, key):
        # GH#42825 enforced in 2.0
        # 创建一个 Series 对象 ser，包含元素 [1, 2]
        ser = Series([1, 2])
        # 使用不支持的字典类型 key 对 ser 进行 loc 操作，预期抛出 TypeError 异常
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            ser.loc[key]

    @pytest.mark.parametrize("key", [{1}, {1: 1}, ({1}, 2), ({1: 1}, 2)])
    def test_getitem_dict_and_set_deprecated_multiindex(self, key):
        # GH#42825 enforced in 2.0
        # 创建一个具有 MultiIndex 索引的 Series 对象 ser，包含元素 [1, 2]
        ser = Series([1, 2], index=MultiIndex.from_tuples([(1, 2), (3, 4)]))
        # 使用不支持的字典类型 key 对 ser 进行 loc 操作，预期抛出 TypeError 异常
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            ser.loc[key]

    @pytest.mark.parametrize("key", [{1}, {1: 1}])
    def test_setitem_dict_and_set_disallowed(self, key):
        # GH#42825 enforced in 2.0
        # 创建一个 Series 对象 ser，包含元素 [1, 2]
        ser = Series([1, 2])
        # 使用不支持的字典类型 key 对 ser 进行 loc 操作赋值，预期抛出 TypeError 异常
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            ser.loc[key] = 1
    @pytest.mark.parametrize("key", [{1}, {1: 1}, ({1}, 2), ({1: 1}, 2)])
    # 使用 pytest 的参数化功能，测试用例参数为不同的键值
    def test_setitem_dict_and_set_disallowed_multiindex(self, key):
        # 在版本2.0中强制执行 GH#42825
        # 创建一个 Series 对象，包含数据 [1, 2]，索引为 MultiIndex 类型的 [(1, 2), (3, 4)]
        ser = Series([1, 2], index=MultiIndex.from_tuples([(1, 2), (3, 4)]))
        # 使用 pytest 断言捕获预期的 TypeError 异常，并验证异常消息是否包含 "as an indexer is not supported"
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            # 尝试使用 ser.loc[key] 的方式进行索引赋值操作，预期会触发 TypeError 异常
            ser.loc[key] = 1
# 测试用例类，用于验证设置项时的有效性
class TestSetitemValidation:
    # 该方法从 pandas/tests/arrays/masked/test_indexing.py 改编而来，用于检查警告而不是错误。
    def _check_setitem_invalid(self, ser, invalid, indexer, warn):
        # 设置警告消息，用于匹配警告信息中的文本
        msg = "Setting an item of incompatible dtype is deprecated"
        msg = re.escape(msg)

        # 复制原始的序列对象，以便在测试中使用
        orig_ser = ser.copy()

        # 使用 assert_produces_warning 上下文，检查设置 ser[indexer] 时是否产生警告
        with tm.assert_produces_warning(warn, match=msg):
            ser[indexer] = invalid
            ser = orig_ser.copy()

        # 使用 assert_produces_warning 上下文，检查设置 ser.iloc[indexer] 时是否产生警告
        with tm.assert_produces_warning(warn, match=msg):
            ser.iloc[indexer] = invalid
            ser = orig_ser.copy()

        # 使用 assert_produces_warning 上下文，检查设置 ser.loc[indexer] 时是否产生警告
        with tm.assert_produces_warning(warn, match=msg):
            ser.loc[indexer] = invalid
            ser = orig_ser.copy()

        # 使用 assert_produces_warning 上下文，检查设置整个序列时是否产生警告
        with tm.assert_produces_warning(warn, match=msg):
            ser[:] = invalid

    # 包含不兼容标量的列表，用于测试
    _invalid_scalars = [
        1 + 2j,
        "True",
        "1",
        "1.0",
        NaT,
        np.datetime64("NaT"),
        np.timedelta64("NaT"),
    ]
    
    # 包含不同索引方式的列表，用于测试
    _indexers = [0, [0], slice(0, 1), [True, False, False], slice(None, None, None)]

    # 使用 pytest 的参数化标记，测试设置布尔类型序列时的标量验证
    @pytest.mark.parametrize(
        "invalid", _invalid_scalars + [1, 1.0, np.int64(1), np.float64(1)]
    )
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_bool(self, invalid, indexer):
        # 创建一个布尔类型的序列
        ser = Series([True, False, False], dtype="bool")
        # 调用 _check_setitem_invalid 方法，验证设置操作是否触发警告
        self._check_setitem_invalid(ser, invalid, indexer, FutureWarning)

    # 使用 pytest 的参数化标记，测试设置整数类型序列时的标量验证
    @pytest.mark.parametrize("invalid", _invalid_scalars + [True, 1.5, np.float64(1.5)])
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_int(self, invalid, any_int_numpy_dtype, indexer):
        # 创建一个整数类型的序列
        ser = Series([1, 2, 3], dtype=any_int_numpy_dtype)
        # 根据不同的无效值来决定是否触发警告
        if isna(invalid) and invalid is not NaT and not np.isnat(invalid):
            warn = None
        else:
            warn = FutureWarning
        # 调用 _check_setitem_invalid 方法，验证设置操作是否触发警告
        self._check_setitem_invalid(ser, invalid, indexer, warn)

    # 使用 pytest 的参数化标记，测试设置浮点数类型序列时的标量验证
    @pytest.mark.parametrize("invalid", _invalid_scalars + [True])
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_float(self, invalid, float_numpy_dtype, indexer):
        # 创建一个浮点数类型的序列
        ser = Series([1, 2, None], dtype=float_numpy_dtype)
        # 调用 _check_setitem_invalid 方法，验证设置操作是否触发警告
        self._check_setitem_invalid(ser, invalid, indexer, FutureWarning)
```