# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_duplicates.py`

```
# 从 itertools 模块导入 product 函数
from itertools import product

# 导入 numpy 库，并简写为 np
import numpy as np

# 导入 pytest 测试框架
import pytest

# 从 pandas._libs 中导入 hashtable 和 libindex
from pandas._libs import (
    hashtable,
    index as libindex,
)

# 从 pandas 库中导入 NA、DatetimeIndex、Index、MultiIndex 和 Series 类
from pandas import (
    NA,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
)

# 导入 pandas._testing 库，并简写为 tm
import pandas._testing as tm


# 定义 pytest fixture：idx_dup，返回一个 MultiIndex 对象
@pytest.fixture
def idx_dup():
    # 定义主索引和次索引
    major_axis = Index(["foo", "bar", "baz", "qux"])
    minor_axis = Index(["one", "two"])

    # 定义主索引和次索引的代码数组
    major_codes = np.array([0, 0, 1, 0, 1, 1])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])

    # 索引的名称列表
    index_names = ["first", "second"]

    # 创建 MultiIndex 对象 mi，包括 levels 和 codes 参数，并关闭完整性验证
    mi = MultiIndex(
        levels=[major_axis, minor_axis],
        codes=[major_codes, minor_codes],
        names=index_names,
        verify_integrity=False,
    )
    return mi


# 定义参数化测试：test_unique，参数 names 分别为 None 和 ["first", "second"]
@pytest.mark.parametrize("names", [None, ["first", "second"]])
def test_unique(names):
    # 使用 from_arrays 方法创建 MultiIndex 对象 mi，设置 names 参数
    mi = MultiIndex.from_arrays([[1, 2, 1, 2], [1, 1, 1, 2]], names=names)

    # 调用 unique 方法，返回结果 res
    res = mi.unique()

    # 使用 from_arrays 方法创建期望的 MultiIndex 对象 exp，设置 names 参数
    exp = MultiIndex.from_arrays([[1, 2, 2], [1, 1, 2]], names=mi.names)

    # 断言 res 和 exp 相等
    tm.assert_index_equal(res, exp)

    # 使用 from_arrays 方法创建 MultiIndex 对象 mi，设置 names 参数
    mi = MultiIndex.from_arrays([list("aaaa"), list("abab")], names=names)

    # 调用 unique 方法，返回结果 res
    res = mi.unique()

    # 使用 from_arrays 方法创建期望的 MultiIndex 对象 exp，设置 names 参数
    exp = MultiIndex.from_arrays([list("aa"), list("ab")], names=mi.names)

    # 断言 res 和 exp 相等
    tm.assert_index_equal(res, exp)

    # 使用 from_arrays 方法创建 MultiIndex 对象 mi，设置 names 参数
    mi = MultiIndex.from_arrays([list("aaaa"), list("aaaa")], names=names)

    # 调用 unique 方法，返回结果 res
    res = mi.unique()

    # 使用 from_arrays 方法创建期望的 MultiIndex 对象 exp，设置 names 参数
    exp = MultiIndex.from_arrays([["a"], ["a"]], names=mi.names)

    # 断言 res 和 exp 相等
    tm.assert_index_equal(res, exp)

    # 创建空的 MultiIndex 对象 mi，设置 names 参数
    mi = MultiIndex.from_arrays([[], []], names=names)

    # 调用 unique 方法，返回结果 res
    res = mi.unique()

    # 断言 mi 和 res 相等
    tm.assert_index_equal(mi, res)


# 定义测试函数：test_unique_datetimelike
def test_unique_datetimelike():
    # 使用 DatetimeIndex 创建 idx1 和 idx2
    idx1 = DatetimeIndex(
        ["2015-01-01", "2015-01-01", "2015-01-01", "2015-01-01", "NaT", "NaT"]
    )
    idx2 = DatetimeIndex(
        ["2015-01-01", "2015-01-01", "2015-01-02", "2015-01-02", "NaT", "2015-01-01"],
        tz="Asia/Tokyo",
    )

    # 调用 from_arrays 方法创建 MultiIndex 对象，设置 names 参数
    result = MultiIndex.from_arrays([idx1, idx2]).unique()

    # 创建期望的 DatetimeIndex 对象 eidx1 和 eidx2，设置 names 参数
    eidx1 = DatetimeIndex(["2015-01-01", "2015-01-01", "NaT", "NaT"])
    eidx2 = DatetimeIndex(
        ["2015-01-01", "2015-01-02", "NaT", "2015-01-01"], tz="Asia/Tokyo"
    )

    # 创建期望的 MultiIndex 对象 exp，设置 names 参数
    exp = MultiIndex.from_arrays([eidx1, eidx2])

    # 断言 result 和 exp 相等
    tm.assert_index_equal(result, exp)


# 定义参数化测试：test_unique_level，参数 level 为 0、"first"、1 和 "second"
@pytest.mark.parametrize("level", [0, "first", 1, "second"])
def test_unique_level(idx, level):
    # 调用 idx 的 unique 方法，设置 level 参数，返回结果 result
    result = idx.unique(level=level)

    # 调用 idx 的 get_level_values 方法获取特定 level 的唯一值，返回期望结果 expected
    expected = idx.get_level_values(level).unique()

    # 断言 result 和 expected 相等
    tm.assert_index_equal(result, expected)

    # 创建 MultiIndex 对象 mi，设置 names 参数
    mi = MultiIndex.from_arrays([[1, 3, 2, 4], [1, 3, 2, 5]], names=["first", "second"])

    # 调用 mi 的 unique 方法，设置 level 参数，返回结果 result
    result = mi.unique(level=level)

    # 调用 mi 的 get_level_values 方法获取特定 level 的值，返回期望结果 expected
    expected = mi.get_level_values(level)

    # 断言 result 和 expected 相等
    tm.assert_index_equal(result, expected)

    # 创建空的 MultiIndex 对象 mi，设置 names 参数
    mi = MultiIndex.from_arrays([[], []], names=["first", "second"])

    # 调用 mi 的 unique 方法，设置 level 参数，返回结果 result
    result = mi.unique(level=level)

    # 调用 mi 的 get_level_values 方法获取特定 level 的值，返回期望结果 expected
    expected = mi.get_level_values(level)

    # 断言 result 和 expected 相等
    tm.assert_index_equal(result, expected)


# 定义测试函数：test_duplicate_multiindex_codes
def test_duplicate_multiindex_codes():
    # GH 17464
    # 此测试用例待补充，未提供完整的代码和注释
    # 确保具有重复级别的 MultiIndex 抛出 ValueError 异常
    msg = r"Level values must be unique: \[[A', ]+\] on level 0"
    # 使用 pytest 来检查是否抛出指定消息的 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        # 创建一个 MultiIndex 对象，其中第一级有 10 个 "A" 和第二级有 0 到 9 的整数
        mi = MultiIndex([["A"] * 10, range(10)], [[0] * 10, range(10)])

    # 确保使用 set_levels 函数设置具有重复级别的 MultiIndex 抛出 ValueError 异常
    # 创建一个 MultiIndex 对象，其中第一级包含 ["A", "A", "B", "B", "B"]，第二级包含 [1, 2, 1, 2, 3]
    mi = MultiIndex.from_arrays([["A", "A", "B", "B", "B"], [1, 2, 1, 2, 3]])
    msg = r"Level values must be unique: \[[AB', ]+\] on level 0"
    # 使用 pytest 来检查是否抛出指定消息的 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        # 使用 set_levels 函数尝试设置第一级为 ["A", "B", "A", "A", "B"] 和第二级为 [2, 1, 3, -2, 5]
        mi.set_levels([["A", "B", "A", "A", "B"], [2, 1, 3, -2, 5]])
@pytest.mark.parametrize("names", [["a", "b", "a"], [1, 1, 2], [1, "a", 1]])
def test_duplicate_level_names(names):
    # 使用 pytest.mark.parametrize 装饰器为 test_duplicate_level_names 函数添加参数化测试
    # 参数 names 包含多个测试用例，每个用例是一个列表，用于测试不同的索引命名情况

    # GH18872, GH19029
    # 创建 MultiIndex 对象 mi，通过 from_product 方法生成多重索引，使用参数 names 作为索引的名称
    mi = MultiIndex.from_product([[0, 1]] * 3, names=names)
    # 断言 mi 对象的 names 属性与参数 names 相等
    assert mi.names == names

    # With .rename()
    # 使用 .rename() 方法重命名 mi 对象的索引
    mi = MultiIndex.from_product([[0, 1]] * 3)
    mi = mi.rename(names)
    # 再次断言 mi 对象的 names 属性与参数 names 相等
    assert mi.names == names

    # With .rename(., level=)
    # 使用 .rename() 方法指定 level 参数来重命名指定层级的索引
    mi.rename(names[1], level=1, inplace=True)
    mi = mi.rename([names[0], names[2]], level=[0, 2])
    # 最后断言 mi 对象的 names 属性与参数 names 相等
    assert mi.names == names


def test_duplicate_meta_data():
    # GH 10115
    # 创建 MultiIndex 对象 mi，设定其 levels 和 codes 属性来测试重复的元数据情况
    mi = MultiIndex(
        levels=[[0, 1], [0, 1, 2]], codes=[[0, 0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 0, 1, 2]]
    )

    # 使用 for 循环测试不同的 MultiIndex 实例 idx
    for idx in [
        mi,
        mi.set_names([None, None]),
        mi.set_names([None, "Num"]),
        mi.set_names(["Upper", "Num"]),
    ]:
        # 断言 idx 对象具有重复项
        assert idx.has_duplicates
        # 断言去除重复项后的 idx 对象的 names 属性与原始 idx 的 names 属性相等
        assert idx.drop_duplicates().names == idx.names


def test_has_duplicates(idx, idx_dup):
    # see fixtures
    # 使用测试夹具 idx 和 idx_dup 来测试 MultiIndex 对象的唯一性和重复性属性
    assert idx.is_unique is True
    assert idx.has_duplicates is False
    assert idx_dup.is_unique is False
    assert idx_dup.has_duplicates is True

    # 创建新的 MultiIndex 对象 mi，测试其唯一性和重复性属性
    mi = MultiIndex(
        levels=[[0, 1], [0, 1, 2]], codes=[[0, 0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 0, 1, 2]]
    )
    assert mi.is_unique is False
    assert mi.has_duplicates is True

    # 创建包含 NaN 的 MultiIndex 对象 mi_nan，测试其唯一性和重复性属性
    mi_nan = MultiIndex(
        levels=[["a", "b"], [0, 1]], codes=[[-1, 0, 0, 1, 1], [-1, 0, 1, 0, 1]]
    )
    assert mi_nan.is_unique is True
    assert mi_nan.has_duplicates is False

    # 创建包含多个 NaN 的 MultiIndex 对象 mi_nan_dup，测试其唯一性和重复性属性
    mi_nan_dup = MultiIndex(
        levels=[["a", "b"], [0, 1]], codes=[[-1, -1, 0, 0, 1, 1], [-1, -1, 0, 1, 0, 1]]
    )
    assert mi_nan_dup.is_unique is False
    assert mi_nan_dup.has_duplicates is True


def test_has_duplicates_from_tuples():
    # GH 9075
    # 创建元组列表 t
    t = [
        ("x", "out", "z", 5, "y", "in", "z", 169),
        ("x", "out", "z", 7, "y", "in", "z", 119),
        ("x", "out", "z", 9, "y", "in", "z", 135),
        ("x", "out", "z", 13, "y", "in", "z", 145),
        ("x", "out", "z", 14, "y", "in", "z", 158),
        ("x", "out", "z", 16, "y", "in", "z", 122),
        ("x", "out", "z", 17, "y", "in", "z", 160),
        ("x", "out", "z", 18, "y", "in", "z", 180),
        ("x", "out", "z", 20, "y", "in", "z", 143),
        ("x", "out", "z", 21, "y", "in", "z", 128),
        ("x", "out", "z", 22, "y", "in", "z", 129),
        ("x", "out", "z", 25, "y", "in", "z", 111),
        ("x", "out", "z", 28, "y", "in", "z", 114),
        ("x", "out", "z", 29, "y", "in", "z", 121),
        ("x", "out", "z", 31, "y", "in", "z", 126),
        ("x", "out", "z", 32, "y", "in", "z", 155),
        ("x", "out", "z", 33, "y", "in", "z", 123),
        ("x", "out", "z", 12, "y", "in", "z", 144),
    ]

    # 从元组列表 t 创建 MultiIndex 对象 mi，测试其是否具有重复项
    mi = MultiIndex.from_tuples(t)
    assert not mi.has_duplicates
# 处理 int64 溢出的可能性
# 如果 nlevels 是 4，则不会溢出
# 如果 nlevels 是 8，则可能会溢出
codes = np.tile(np.arange(500), 2)  # 创建一个长度为 1000 的数组，内容为 [0, 1, ..., 499, 0, 1, ..., 499]
level = np.arange(500)  # 创建一个长度为 500 的数组，内容为 [0, 1, 2, ..., 499]

if with_nulls:  # 如果 with_nulls 为 True，则插入一些空值
    codes[500] = -1  # 常见的 NaN 值
    codes = [codes.copy() for i in range(nlevels)]  # 复制数组 codes，共 nlevels 份
    for i in range(nlevels):
        codes[i][500 + i - nlevels // 2] = -1  # 在每份复制的数组中插入空值

    codes += [np.array([-1, 1]).repeat(500)]  # 添加一个新的包含 -1 和 1 重复的数组到 codes 中
else:
    codes = [codes] * nlevels + [np.arange(2).repeat(500)]  # 将数组 codes 复制 nlevels 次，然后添加一个重复的 [0, 1] 数组

levels = [level] * nlevels + [[0, 1]]  # 将数组 level 复制 nlevels 次，然后添加一个 [0, 1] 数组

# 创建一个 MultiIndex 对象 mi，使用 levels 和 codes
mi = MultiIndex(levels=levels, codes=codes)
assert not mi.has_duplicates  # 断言 mi 中没有重复值

# 如果 with_nulls 为 True，则进入条件
if with_nulls:

    def f(a):
        return np.insert(a, 1000, a[0])

    codes = list(map(f, codes))  # 对 codes 中的每个数组应用函数 f
    mi = MultiIndex(levels=levels, codes=codes)  # 使用新的 codes 创建 MultiIndex 对象
else:
    values = mi.values.tolist()  # 将 mi 的值转换为列表
    mi = MultiIndex.from_tuples(values + [values[0]])  # 从元组列表创建 MultiIndex 对象，包含一个重复的第一个值

assert mi.has_duplicates  # 断言 mi 中有重复值


@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", [False, False, False, True, True, False]),
        ("last", [False, True, True, False, False, False]),
        (False, [False, True, True, True, True, False]),
    ],
)
def test_duplicated(idx_dup, keep, expected):
    result = idx_dup.duplicated(keep=keep)  # 调用 duplicated 方法计算重复值
    expected = np.array(expected)  # 创建预期的结果数组
    tm.assert_numpy_array_equal(result, expected)  # 断言计算结果与预期结果相等


@pytest.mark.arm_slow
def test_duplicated_hashtable_impl(keep, monkeypatch):
    # GH 9125
    n, k = 6, 10
    levels = [np.arange(n), [str(i) for i in range(n)], 1000 + np.arange(n)]
    codes = [np.random.default_rng(2).choice(n, k * n) for _ in levels]

    with monkeypatch.context() as m:
        m.setattr(libindex, "_SIZE_CUTOFF", 50)  # 使用 monkeypatch 设置 libindex 的 _SIZE_CUTOFF 属性为 50
        mi = MultiIndex(levels=levels, codes=codes)  # 创建 MultiIndex 对象

        result = mi.duplicated(keep=keep)  # 调用 duplicated 方法计算重复值
        expected = hashtable.duplicated(mi.values, keep=keep)  # 使用 hashtable 模块计算预期结果

    tm.assert_numpy_array_equal(result, expected)  # 断言计算结果与预期结果相等


@pytest.mark.parametrize("val", [101, 102])
def test_duplicated_with_nan(val):
    # GH5873
    mi = MultiIndex.from_arrays([[101, val], [3.5, np.nan]])  # 使用数组创建 MultiIndex 对象
    assert not mi.has_duplicates  # 断言 mi 中没有重复值

    tm.assert_numpy_array_equal(mi.duplicated(), np.zeros(2, dtype="bool"))  # 断言计算结果与预期结果相等


@pytest.mark.parametrize("n", range(1, 6))
@pytest.mark.parametrize("m", range(1, 5))
def test_duplicated_with_nan_multi_shape(n, m):
    # GH5873
    # 包括 NaN 在内的所有可能的唯一组合
    codes = product(range(-1, n), range(-1, m))
    mi = MultiIndex(
        levels=[list("abcde")[:n], list("WXYZ")[:m]],  # 使用列表创建 levels
        codes=np.random.default_rng(2).permutation(list(codes)).T,  # 使用随机排列的 codes 创建 MultiIndex 对象
    )
    assert len(mi) == (n + 1) * (m + 1)  # 断言 mi 的长度符合预期
    assert not mi.has_duplicates  # 断言 mi 中没有重复值

    tm.assert_numpy_array_equal(mi.duplicated(), np.zeros(len(mi), dtype="bool"))  # 断言计算结果与预期结果相等


def test_duplicated_drop_duplicates():
    # GH#4060
    idx = MultiIndex.from_arrays(([1, 2, 3, 1, 2, 3], [1, 1, 1, 1, 2, 2]))  # 使用数组创建 MultiIndex 对象
    # 创建一个预期的布尔类型的 NumPy 数组，表示重复值的预期结果
    expected = np.array([False, False, False, True, False, False], dtype=bool)
    # 使用 idx 对象的 duplicated() 方法找到重复项并返回布尔数组
    duplicated = idx.duplicated()
    # 使用测试工具确保得到的 duplicated 数组与预期数组相等
    tm.assert_numpy_array_equal(duplicated, expected)
    # 断言检查 duplicated 数组的数据类型是否为布尔型
    assert duplicated.dtype == bool
    # 创建一个预期的 MultiIndex 对象，使用 from_arrays 方法构造
    expected = MultiIndex.from_arrays(([1, 2, 3, 2, 3], [1, 1, 1, 2, 2]))
    # 使用测试工具确保 idx 对象调用 drop_duplicates() 后的结果与预期的 MultiIndex 对象相等
    tm.assert_index_equal(idx.drop_duplicates(), expected)

    # 创建另一个预期的布尔类型的 NumPy 数组，表示使用 keep="last" 参数时的预期结果
    expected = np.array([True, False, False, False, False, False])
    # 使用 idx 对象的 duplicated(keep="last") 方法找到重复项并返回布尔数组
    duplicated = idx.duplicated(keep="last")
    # 使用测试工具确保得到的 duplicated 数组与预期数组相等
    tm.assert_numpy_array_equal(duplicated, expected)
    # 断言检查 duplicated 数组的数据类型是否为布尔型
    assert duplicated.dtype == bool
    # 创建另一个预期的 MultiIndex 对象，使用 from_arrays 方法构造
    expected = MultiIndex.from_arrays(([2, 3, 1, 2, 3], [1, 1, 1, 2, 2]))
    # 使用测试工具确保 idx 对象调用 drop_duplicates(keep="last") 后的结果与预期的 MultiIndex 对象相等
    tm.assert_index_equal(idx.drop_duplicates(keep="last"), expected)

    # 创建另一个预期的布尔类型的 NumPy 数组，表示使用 keep=False 参数时的预期结果
    expected = np.array([True, False, False, True, False, False])
    # 使用 idx 对象的 duplicated(keep=False) 方法找到重复项并返回布尔数组
    duplicated = idx.duplicated(keep=False)
    # 使用测试工具确保得到的 duplicated 数组与预期数组相等
    tm.assert_numpy_array_equal(duplicated, expected)
    # 断言检查 duplicated 数组的数据类型是否为布尔型
    assert duplicated.dtype == bool
    # 创建另一个预期的 MultiIndex 对象，使用 from_arrays 方法构造
    expected = MultiIndex.from_arrays(([2, 3, 2, 3], [1, 1, 2, 2]))
    # 使用测试工具确保 idx 对象调用 drop_duplicates(keep=False) 后的结果与预期的 MultiIndex 对象相等
    tm.assert_index_equal(idx.drop_duplicates(keep=False), expected)
# 测试复数数据类型的重复序列情况
def test_duplicated_series_complex_numbers(complex_dtype):
    # GH 17927：GitHub上的issue编号
    expected = Series(
        [False, False, False, True, False, False, False, True, False, True],
        dtype=bool,
    )
    # 创建一个包含复数的Series，检测其中的重复项
    result = Series(
        [
            np.nan + np.nan * 1j,  # 复数 NaN + NaNj
            0,                     # 实数 0
            1j,                    # 纯虚数 j
            1j,                    # 纯虚数 j，重复
            1,                     # 实数 1
            1 + 1j,                # 复数 1 + j
            1 + 2j,                # 复数 1 + 2j
            1 + 1j,                # 复数 1 + j，重复
            np.nan,                # NaN
            np.nan + np.nan * 1j,  # 复数 NaN + NaNj，重复
        ],
        dtype=complex_dtype,       # 指定复数数据类型
    ).duplicated()
    # 断言检查结果是否与预期相等
    tm.assert_series_equal(result, expected)


# 测试多级索引中唯一值的数据类型
def test_midx_unique_ea_dtype():
    # GH#48335：GitHub上的issue编号
    vals_a = Series([1, 2, NA, NA], dtype="Int64")
    vals_b = np.array([1, 2, 3, 3])
    # 创建一个包含多级索引的MultiIndex，从两个数组创建，指定名称为"a"和"b"
    midx = MultiIndex.from_arrays([vals_a, vals_b], names=["a", "b"])
    # 获取多级索引中的唯一值
    result = midx.unique()

    exp_vals_a = Series([1, 2, NA], dtype="Int64")
    exp_vals_b = np.array([1, 2, 3])
    # 创建预期的多级索引对象，与上述方法相同
    expected = MultiIndex.from_arrays([exp_vals_a, exp_vals_b], names=["a", "b"])
    # 断言检查结果是否与预期相等
    tm.assert_index_equal(result, expected)
```