# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_sorting.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

from pandas.errors import UnsortedIndexError  # 从 pandas.errors 中导入 UnsortedIndexError 异常类

from pandas import (  # 从 pandas 库中导入以下对象：
    CategoricalIndex,  # 分类索引
    DataFrame,  # 数据帧
    Index,  # 索引
    MultiIndex,  # 多重索引
    RangeIndex,  # 范围索引
    Series,  # 系列
    Timestamp,  # 时间戳
)
import pandas._testing as tm  # 导入 pandas._testing 模块，并使用 tm 别名

from pandas.core.indexes.frozen import FrozenList  # 从 pandas.core.indexes.frozen 导入 FrozenList 类


def test_sortlevel(idx):
    tuples = list(idx)  # 将索引对象转换为元组列表
    np.random.default_rng(2).shuffle(tuples)  # 使用随机种子2对元组列表进行洗牌操作

    index = MultiIndex.from_tuples(tuples)  # 使用洗牌后的元组列表创建多重索引对象

    sorted_idx, _ = index.sortlevel(0)  # 对索引的第0级别进行排序
    expected = MultiIndex.from_tuples(sorted(tuples))  # 创建期望的排序后的多重索引对象
    assert sorted_idx.equals(expected)  # 断言排序后的索引与期望相同

    sorted_idx, _ = index.sortlevel(0, ascending=False)  # 对索引的第0级别进行降序排序
    assert sorted_idx.equals(expected[::-1])  # 断言排序后的索引与期望的逆序相同

    sorted_idx, _ = index.sortlevel(1)  # 对索引的第1级别进行排序
    by1 = sorted(tuples, key=lambda x: (x[1], x[0]))  # 按照第1级别和第0级别的顺序对元组列表进行排序
    expected = MultiIndex.from_tuples(by1)  # 创建期望的排序后的多重索引对象
    assert sorted_idx.equals(expected)  # 断言排序后的索引与期望相同

    sorted_idx, _ = index.sortlevel(1, ascending=False)  # 对索引的第1级别进行降序排序
    assert sorted_idx.equals(expected[::-1])  # 断言排序后的索引与期望的逆序相同


def test_sortlevel_not_sort_remaining():
    mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))  # 创建具有名称 "A", "B", "C" 的多重索引对象
    sorted_idx, _ = mi.sortlevel("A", sort_remaining=False)  # 按照级别 "A" 进行排序，但不对剩余级别进行排序
    assert sorted_idx.equals(mi)  # 断言排序后的索引与原始索引相同


def test_sortlevel_deterministic():
    tuples = [  # 定义元组列表
        ("bar", "one"),
        ("foo", "two"),
        ("qux", "two"),
        ("foo", "one"),
        ("baz", "two"),
        ("qux", "one"),
    ]

    index = MultiIndex.from_tuples(tuples)  # 使用元组列表创建多重索引对象

    sorted_idx, _ = index.sortlevel(0)  # 对索引的第0级别进行排序
    expected = MultiIndex.from_tuples(sorted(tuples))  # 创建期望的排序后的多重索引对象
    assert sorted_idx.equals(expected)  # 断言排序后的索引与期望相同

    sorted_idx, _ = index.sortlevel(0, ascending=False)  # 对索引的第0级别进行降序排序
    assert sorted_idx.equals(expected[::-1])  # 断言排序后的索引与期望的逆序相同

    sorted_idx, _ = index.sortlevel(1)  # 对索引的第1级别进行排序
    by1 = sorted(tuples, key=lambda x: (x[1], x[0]))  # 按照第1级别和第0级别的顺序对元组列表进行排序
    expected = MultiIndex.from_tuples(by1)  # 创建期望的排序后的多重索引对象
    assert sorted_idx.equals(expected)  # 断言排序后的索引与期望相同

    sorted_idx, _ = index.sortlevel(1, ascending=False)  # 对索引的第1级别进行降序排序
    assert sorted_idx.equals(expected[::-1])  # 断言排序后的索引与期望的逆序相同


def test_sortlevel_na_position():
    # GH#51612
    midx = MultiIndex.from_tuples([(1, np.nan), (1, 1)])  # 创建具有 NaN 值的多重索引对象
    result = midx.sortlevel(level=[0, 1], na_position="last")[0]  # 按照指定级别进行排序，NaN 值放在最后
    expected = MultiIndex.from_tuples([(1, 1), (1, np.nan)])  # 创建期望的排序后的多重索引对象
    tm.assert_index_equal(result, expected)  # 使用 pandas._testing 模块中的方法断言索引对象相等


def test_numpy_argsort(idx):
    result = np.argsort(idx)  # 使用 NumPy 对索引对象进行排序
    expected = idx.argsort()  # 获取索引对象的排序后的索引数组
    tm.assert_numpy_array_equal(result, expected)  # 使用 pandas._testing 模块中的方法断言 NumPy 数组相等

    # these are the only two types that perform
    # pandas compatibility input validation - the
    # rest already perform separate (or no) such
    # validation via their 'values' attribute as
    # defined in pandas.core.indexes/base.py - they
    # cannot be changed at the moment due to
    # backwards compatibility concerns
    # 如果 idx 的类型是 CategoricalIndex 或者 RangeIndex，则执行以下代码块
    if isinstance(type(idx), (CategoricalIndex, RangeIndex)):
        # 设置错误消息，指出 'axis' 参数不被支持
        msg = "the 'axis' parameter is not supported"
        # 使用 pytest 来检测是否会抛出 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            # 调用 numpy 的 argsort 函数，预期会抛出 ValueError 异常
            np.argsort(idx, axis=1)
    
        # 设置错误消息，指出 'kind' 参数不被支持
        msg = "the 'kind' parameter is not supported"
        # 使用 pytest 来检测是否会抛出 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            # 调用 numpy 的 argsort 函数，预期会抛出 ValueError 异常
            np.argsort(idx, kind="mergesort")
    
        # 设置错误消息，指出 'order' 参数不被支持
        msg = "the 'order' parameter is not supported"
        # 使用 pytest 来检测是否会抛出 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            # 调用 numpy 的 argsort 函数，预期会抛出 ValueError 异常
            np.argsort(idx, order=("a", "b"))
def test_unsortedindex():
    # GH 11897: 测试用例编号，用于参考相关问题
    mi = MultiIndex.from_tuples(
        [("z", "a"), ("x", "a"), ("y", "b"), ("x", "b"), ("y", "a"), ("z", "b")],
        names=["one", "two"],
    )
    # 创建一个带有多级索引的 DataFrame
    df = DataFrame([[i, 10 * i] for i in range(6)], index=mi, columns=["one", "two"])

    # GH 16734: 不是排序的，但没有真正的切片操作
    # 使用 loc 方法进行多级索引切片，选择 ("z", "a") 对应的行
    result = df.loc(axis=0)["z", "a"]
    # 期望结果为 df 的第一行
    expected = df.iloc[0]
    # 使用测试工具函数比较两个 Series 是否相等
    tm.assert_series_equal(result, expected)

    # 准备一个错误信息，测试切片时要求索引必须是 lexsorted
    msg = (
        "MultiIndex slicing requires the index to be lexsorted: "
        r"slicing on levels \[1\], lexsort depth 0"
    )
    # 用 pytest 来断言是否会抛出 UnsortedIndexError 异常，并且异常信息匹配特定的消息
    with pytest.raises(UnsortedIndexError, match=msg):
        df.loc(axis=0)["z", slice("a")]

    # 对 DataFrame 进行排序，按索引进行排序
    df.sort_index(inplace=True)
    # 断言经过排序后，选择多级索引 ("z", :) 的结果行数为 2
    assert len(df.loc(axis=0)["z", :]) == 2

    # 使用 pytest 断言是否会抛出 KeyError 异常，并且异常信息匹配 "'q'"
    with pytest.raises(KeyError, match="'q'"):
        df.loc(axis=0)["q", :]


def test_unsortedindex_doc_examples(performance_warning):
    # https://pandas.pydata.org/pandas-docs/stable/advanced.html#sorting-a-multiindex
    # 创建一个带有多级索引的 DataFrame
    dfm = DataFrame(
        {
            "jim": [0, 0, 1, 1],
            "joe": ["x", "x", "z", "y"],
            "jolie": np.random.default_rng(2).random(4),
        }
    )
    # 设置 DataFrame 的多级索引
    dfm = dfm.set_index(["jim", "joe"])
    # 使用测试工具函数来检查是否会产生性能警告
    with tm.assert_produces_warning(performance_warning):
        dfm.loc[(1, "z")]

    # 准备一个错误信息，测试切片时要求索引必须是 lexsorted
    msg = r"Key length \(2\) was greater than MultiIndex lexsort depth \(1\)"
    # 使用 pytest 来断言是否会抛出 UnsortedIndexError 异常，并且异常信息匹配特定的消息
    with pytest.raises(UnsortedIndexError, match=msg):
        dfm.loc[(0, "y") : (1, "z")]

    # 断言 DataFrame 的索引并非 lexsorted
    assert not dfm.index._is_lexsorted()
    # 断言 DataFrame 的索引的 lexsort 深度为 1
    assert dfm.index._lexsort_depth == 1

    # 对 DataFrame 进行排序，按索引进行排序
    dfm = dfm.sort_index()
    dfm.loc[(1, "z")]
    dfm.loc[(0, "y") : (1, "z")]

    # 断言 DataFrame 的索引已经是 lexsorted
    assert dfm.index._is_lexsorted()
    # 断言 DataFrame 的索引的 lexsort 深度为 2


def test_reconstruct_sort():
    # starts off lexsorted & monotonic
    # 创建一个初始为 lexsorted 和 monotonic 的 MultiIndex 对象
    mi = MultiIndex.from_arrays([["A", "A", "B", "B", "B"], [1, 2, 1, 2, 3]])
    # 断言 MultiIndex 是否是单调递增的
    assert mi.is_monotonic_increasing
    # 使用 _sort_levels_monotonic 方法重建 MultiIndex 对象
    recons = mi._sort_levels_monotonic()
    # 断言重建后的 MultiIndex 是否是单调递增的
    assert recons.is_monotonic_increasing
    # 断言原始的 mi 和重建后的 recons 是同一个对象
    assert mi is recons

    # 断言 mi 和 recons 是否相等
    assert mi.equals(recons)
    # 断言两个 Index 对象是否相等
    assert Index(mi.values).equals(Index(recons.values))

    # 不能转换为 lexsorted
    # 创建一个初始未排序的 MultiIndex 对象
    mi = MultiIndex.from_tuples(
        [("z", "a"), ("x", "a"), ("y", "b"), ("x", "b"), ("y", "a"), ("z", "b")],
        names=["one", "two"],
    )
    # 断言 mi 不是单调递增的
    assert not mi.is_monotonic_increasing
    # 使用 _sort_levels_monotonic 方法重建 MultiIndex 对象
    recons = mi._sort_levels_monotonic()
    # 断言重建后的 MultiIndex 仍然不是单调递增的
    assert not recons.is_monotonic_increasing
    # 断言 mi 和 recons 是否相等
    assert mi.equals(recons)
    # 断言两个 Index 对象是否相等
    assert Index(mi.values).equals(Index(recons.values))

    # 不能转换为 lexsorted
    # 创建一个初始未排序的 MultiIndex 对象
    mi = MultiIndex(
        levels=[["b", "d", "a"], [1, 2, 3]],
        codes=[[0, 1, 0, 2], [2, 0, 0, 1]],
        names=["col1", "col2"],
    )
    # 断言 mi 不是单调递增的
    assert not mi.is_monotonic_increasing
    # 使用 _sort_levels_monotonic 方法重建 MultiIndex 对象
    recons = mi._sort_levels_monotonic()
    # 断言重建后的 MultiIndex 仍然不是单调递增的
    assert not recons.is_monotonic_increasing
    # 断言 mi 和 recons 是否相等
    assert mi.equals(recons)
    # 断言两个 Index 对象是否相等
    assert Index(mi.values).equals(Index(recons.values))


def test_reconstruct_remove_unused():
    # xref to GH 2770
    # 创建一个 DataFrame 对象，包含三行数据和三列：'first', 'second', 'third'
    df = DataFrame(
        [["deleteMe", 1, 9], ["keepMe", 2, 9], ["keepMeToo", 3, 9]],
        columns=["first", "second", "third"],
    )
    # 将 ('first', 'second') 设为索引，保留原索引列
    df2 = df.set_index(["first", "second"], drop=False)
    # 从 df2 中筛选出 'first' 列不等于 "deleteMe" 的行
    df2 = df2[df2["first"] != "deleteMe"]

    # 创建一个预期的 MultiIndex 对象，包含两个级别和对应的索引值
    expected = MultiIndex(
        levels=[["deleteMe", "keepMe", "keepMeToo"], [1, 2, 3]],
        codes=[[1, 2], [1, 2]],
        names=["first", "second"],
    )
    # 将 df2 的索引存储到 result 中
    result = df2.index
    # 断言 result 等于预期的 MultiIndex 对象 expected
    tm.assert_index_equal(result, expected)

    # 创建另一个预期的 MultiIndex 对象，去除未使用的级别
    expected = MultiIndex(
        levels=[["keepMe", "keepMeToo"], [2, 3]],
        codes=[[0, 1], [0, 1]],
        names=["first", "second"],
    )
    # 对 df2 的索引应用 remove_unused_levels() 方法，并存储结果到 result 中
    result = df2.index.remove_unused_levels()
    # 断言 result 等于预期的 MultiIndex 对象 expected
    tm.assert_index_equal(result, expected)

    # 对已移除未使用级别的索引再次应用 remove_unused_levels() 方法，预期结果不变
    result2 = result.remove_unused_levels()
    # 断言 result2 等于 result，即 remove_unused_levels() 方法是幂等的
    tm.assert_index_equal(result2, expected)
    # 使用 assert 来确保 result2 与 result 是同一个对象
    assert result2 is result
@pytest.mark.parametrize(
    "first_type,second_type", [("int64", "int64"), ("datetime64[D]", "str")]
)
# 定义测试函数 test_remove_unused_levels_large，参数化两种数据类型
def test_remove_unused_levels_large(first_type, second_type):
    # GH16556
    # 因为测试应该是确定性的（特别是这个测试检查是否删除了层级，对于每个随机输入情况并非如此）：
    
    # 设置随机数生成器并指定种子值（这个种子值是一个有效的任意值）
    rng = np.random.default_rng(10)

    # 创建一个数据框 DataFrame
    size = 1 << 16  # 计算大小为 2^16
    df = DataFrame(
        {
            "first": rng.integers(0, 1 << 13, size).astype(first_type),
            "second": rng.integers(0, 1 << 10, size).astype(second_type),
            "third": rng.random(size),
        }
    )
    # 根据 'first' 和 'second' 列进行分组求和
    df = df.groupby(["first", "second"]).sum()
    # 选择满足条件 'third < 0.1' 的行
    df = df[df.third < 0.1]

    # 调用 remove_unused_levels 方法
    result = df.index.remove_unused_levels()
    # 检查第一个层级的长度是否小于原始索引的第一个层级长度
    assert len(result.levels[0]) < len(df.index.levels[0])
    # 检查第二个层级的长度是否小于原始索引的第二个层级长度
    assert len(result.levels[1]) < len(df.index.levels[1])
    # 检查结果是否与原始索引相等
    assert result.equals(df.index)

    # 重新设置索引并设为期望值
    expected = df.reset_index().set_index(["first", "second"]).index
    # 使用 assert_index_equal 检查结果是否与期望值相等
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("level0", [["a", "d", "b"], ["a", "d", "b", "unused"]])
@pytest.mark.parametrize(
    "level1", [["w", "x", "y", "z"], ["w", "x", "y", "z", "unused"]]
)
# 定义测试函数 test_remove_unused_nan，参数化两个级别的列表
def test_remove_unused_nan(level0, level1):
    # GH 18417
    
    # 创建一个 MultiIndex 对象
    mi = MultiIndex(levels=[level0, level1], codes=[[0, 2, -1, 1, -1], [0, 1, 2, 3, 2]])

    # 调用 remove_unused_levels 方法
    result = mi.remove_unused_levels()
    # 使用 assert_index_equal 检查结果是否与原始 MultiIndex 对象相等
    tm.assert_index_equal(result, mi)
    # 遍历每个级别，确保结果中不包含 "unused"
    for level in 0, 1:
        assert "unused" not in result.levels[level]


# 定义测试函数 test_argsort，接受一个参数 idx
def test_argsort(idx):
    # 调用 argsort 方法
    result = idx.argsort()
    # 获取预期结果
    expected = idx.values.argsort()
    # 使用 assert_numpy_array_equal 检查结果是否与预期一致
    tm.assert_numpy_array_equal(result, expected)


def test_remove_unused_levels_with_nan():
    # GH 37510
    
    # 创建一个带有 NaN 的 Index 对象
    idx = Index([(1, np.nan), (3, 4)]).rename(["id1", "id2"])
    # 设置 'id1' 级别的值
    idx = idx.set_levels(["a", np.nan], level="id1")
    # 调用 remove_unused_levels 方法
    idx = idx.remove_unused_levels()
    # 获取结果的 levels
    result = idx.levels
    # 设置预期的 levels
    expected = FrozenList([["a", np.nan], [4]])
    # 检查结果的字符串表示是否与预期一致
    assert str(result) == str(expected)


def test_sort_values_nan():
    # GH48495, GH48626
    
    # 创建一个 MultiIndex 对象
    midx = MultiIndex(levels=[["A", "B", "C"], ["D"]], codes=[[1, 0, 2], [-1, -1, 0]])
    # 调用 sort_values 方法
    result = midx.sort_values()
    # 设置预期的 MultiIndex 对象
    expected = MultiIndex(
        levels=[["A", "B", "C"], ["D"]], codes=[[0, 1, 2], [-1, -1, 0]]
    )
    # 使用 assert_index_equal 检查结果是否与预期一致
    tm.assert_index_equal(result, expected)


def test_sort_values_incomparable():
    # GH48495
    
    # 从数组创建一个 MultiIndex 对象
    mi = MultiIndex.from_arrays(
        [
            [1, Timestamp("2000-01-01")],
            [3, 4],
        ]
    )
    # 设置错误匹配信息
    match = "'<' not supported between instances of 'Timestamp' and 'int'"
    # 使用 pytest.raises 检查是否抛出预期的 TypeError 异常
    with pytest.raises(TypeError, match=match):
        mi.sort_values()


@pytest.mark.parametrize("na_position", ["first", "last"])
@pytest.mark.parametrize("dtype", ["float64", "Int64", "Float64"])
# 定义测试函数 test_sort_values_with_na_na_position，参数化两个参数
def test_sort_values_with_na_na_position(dtype, na_position):
    # 51612
    
    # 创建包含两个 Series 的数组
    arrays = [
        Series([1, 1, 2], dtype=dtype),
        Series([1, None, 3], dtype=dtype),
    ]
    # 使用这些数组创建一个 MultiIndex 对象
    index = MultiIndex.from_arrays(arrays)
    # 对索引进行排序，并将结果赋给变量 result
    result = index.sort_values(na_position=na_position)
    
    # 检查 na_position 变量的取值，如果为 "first" 则执行以下代码块
    if na_position == "first":
        # 创建包含两个 Series 对象的列表 arrays，这些 Series 对象分别包含指定的数据和空值 None
        arrays = [
            Series([1, 1, 2], dtype=dtype),  # 第一个 Series 包含数值 1, 1, 2
            Series([None, 1, 3], dtype=dtype),  # 第二个 Series 包含一个 None 和数值 1, 3
        ]
    # 如果 na_position 不为 "first"，则执行以下代码块
    else:
        # 创建包含两个 Series 对象的列表 arrays，这些 Series 对象分别包含指定的数据和空值 None
        arrays = [
            Series([1, 1, 2], dtype=dtype),  # 第一个 Series 包含数值 1, 1, 2
            Series([1, None, 3], dtype=dtype),  # 第二个 Series 包含数值 1 和一个 None, 3
        ]
    
    # 使用 arrays 列表创建一个期望的 MultiIndex 对象，并将结果赋给变量 expected
    expected = MultiIndex.from_arrays(arrays)
    
    # 使用测试工具 tm.assert_index_equal 检查 result 是否等于 expected
    tm.assert_index_equal(result, expected)
# 定义名为 test_sort_unnecessary_warning 的测试函数，用于测试某个特定问题的排序功能
def test_sort_unnecessary_warning():
    # 创建一个多级索引对象 midx，包含三个元组 (1.5, 2), (3.5, 3), (0, 1)
    midx = MultiIndex.from_tuples([(1.5, 2), (3.5, 3), (0, 1)])
    # 将 midx 对象的第一级别的索引值设置为 [2.5, NaN, 1]
    midx = midx.set_levels([2.5, np.nan, 1], level=0)
    # 对 midx 进行排序，并将结果赋给 result
    result = midx.sort_values()
    # 创建预期的多级索引对象 expected，包含三个元组 (1, 3), (2.5, 1), (NaN, 2)
    expected = MultiIndex.from_tuples([(1, 3), (2.5, 1), (np.nan, 2)])
    # 断言 result 与 expected 相等，即检查排序后的结果是否符合预期
    tm.assert_index_equal(result, expected)
```