# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_join.py`

```
import numpy as np  # 导入 NumPy 库，用于数组操作
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入以下模块：
    DataFrame,  # 数据帧，用于存储二维标记数据
    Index,  # 索引对象，用于标签索引
    Interval,  # 区间对象，用于表示间隔或区间索引
    MultiIndex,  # 多重索引对象，用于多级索引
    Series,  # 系列对象，用于带标签的一维数组
    StringDtype,  # 字符串类型，用于处理字符串数据
)
import pandas._testing as tm  # 导入 pandas 内部的测试工具模块


@pytest.mark.parametrize("other", [["three", "one", "two"], ["one"], ["one", "three"]])
def test_join_level(idx, other, join_type):
    other = Index(other)  # 将列表 other 转换为 Index 对象
    join_index, lidx, ridx = other.join(  # 执行索引对象的 join 操作，返回连接后的索引以及左右索引器
        idx, how=join_type, level="second", return_indexers=True
    )

    exp_level = other.join(idx.levels[1], how=join_type)  # 根据指定级别执行 join 操作
    assert join_index.levels[0].equals(idx.levels[0])  # 检查连接后的索引第一级是否相等
    assert join_index.levels[1].equals(exp_level)  # 检查连接后的索引第二级是否与预期相等

    # pare down levels
    mask = np.array([x[1] in exp_level for x in idx], dtype=bool)  # 创建布尔掩码以筛选符合条件的索引值
    exp_values = idx.values[mask]  # 根据掩码选择符合条件的索引值
    tm.assert_numpy_array_equal(join_index.values, exp_values)  # 检查连接后的索引值是否与预期相等

    if join_type in ("outer", "inner"):
        join_index2, ridx2, lidx2 = idx.join(  # 执行另一种 join 操作
            other, how=join_type, level="second", return_indexers=True
        )

        assert join_index.equals(join_index2)  # 检查两次 join 的结果是否一致
        tm.assert_numpy_array_equal(lidx, lidx2)  # 检查左索引器是否一致
        if ridx is None:
            assert ridx == ridx2  # 如果右索引器为空，检查其是否相等
        else:
            tm.assert_numpy_array_equal(ridx, ridx2)  # 否则，检查右索引器是否一致
        tm.assert_numpy_array_equal(join_index2.values, exp_values)  # 检查第二次 join 的索引值是否与预期相等


def test_join_level_corner_case(idx):
    # some corner cases
    index = Index(["three", "one", "two"])  # 创建一个新的索引对象
    result = index.join(idx, level="second")  # 执行索引对象的 join 操作
    assert isinstance(result, MultiIndex)  # 检查返回的结果是否为 MultiIndex 对象

    with pytest.raises(TypeError, match="Join.*MultiIndex.*ambiguous"):
        idx.join(idx, level=1)  # 检查在特定情况下是否会引发 TypeError 异常


def test_join_self(idx, join_type):
    result = idx.join(idx, how=join_type)  # 执行索引对象与自身的 join 操作
    expected = idx  # 将预期结果设置为输入的索引对象
    if join_type == "outer":
        expected = expected.sort_values()  # 如果是 outer join，则对预期结果进行排序
    tm.assert_index_equal(result, expected)  # 检查实际结果与预期结果是否相等


def test_join_multi():
    # GH 10665
    midx = MultiIndex.from_product([np.arange(4), np.arange(4)], names=["a", "b"])  # 创建一个多重索引对象
    idx = Index([1, 2, 5], name="b")  # 创建一个新的索引对象

    # inner
    jidx, lidx, ridx = midx.join(idx, how="inner", return_indexers=True)  # 执行 inner join 操作
    exp_idx = MultiIndex.from_product([np.arange(4), [1, 2]], names=["a", "b"])  # 预期的结果索引对象
    exp_lidx = np.array([1, 2, 5, 6, 9, 10, 13, 14], dtype=np.intp)  # 预期的左索引器数组
    exp_ridx = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.intp)  # 预期的右索引器数组
    tm.assert_index_equal(jidx, exp_idx)  # 检查 join 后的索引是否与预期相等
    tm.assert_numpy_array_equal(lidx, exp_lidx)  # 检查左索引器是否与预期相等
    tm.assert_numpy_array_equal(ridx, exp_ridx)  # 检查右索引器是否与预期相等

    # flip
    jidx, ridx, lidx = idx.join(midx, how="inner", return_indexers=True)  # 执行 inner join 操作（反转顺序）
    tm.assert_index_equal(jidx, exp_idx)  # 检查 join 后的索引是否与预期相等
    tm.assert_numpy_array_equal(lidx, exp_lidx)  # 检查左索引器是否与预期相等
    tm.assert_numpy_array_equal(ridx, exp_ridx)  # 检查右索引器是否与预期相等

    # keep MultiIndex
    jidx, lidx, ridx = midx.join(idx, how="left", return_indexers=True)  # 执行 left join 操作
    exp_ridx = np.array(
        [-1, 0, 1, -1, -1, 0, 1, -1, -1, 0, 1, -1, -1, 0, 1, -1], dtype=np.intp
    )  # 预期的右索引器数组
    tm.assert_index_equal(jidx, midx)  # 检查 join 后的索引是否与预期相等
    assert lidx is None  # 检查左索引器是否为空
    tm.assert_numpy_array_equal(ridx, exp_ridx)  # 检查右索引器是否与预期相等

    # flip
    jidx, ridx, lidx = idx.join(midx, how="right", return_indexers=True)  # 执行 right join 操作
    # 使用测试框架中的函数检查 jidx 和 midx 是否索引相等
    tm.assert_index_equal(jidx, midx)
    # 使用断言检查 lidx 是否为 None，如果不是则会触发 AssertionError
    assert lidx is None
    # 使用测试框架中的函数检查 ridx 和 exp_ridx 是否对应的 NumPy 数组相等
    tm.assert_numpy_array_equal(ridx, exp_ridx)
def test_join_multi_wrong_order():
    # 定义一个测试函数，用于验证 MultiIndex 对象的 join 方法在索引顺序错误时的行为
    # GH 25760
    # GH 28956

    # 创建两个 MultiIndex 对象，分别表示索引顺序为 ["a", "b"] 和 ["b", "a"]
    midx1 = MultiIndex.from_product([[1, 2], [3, 4]], names=["a", "b"])
    midx2 = MultiIndex.from_product([[1, 2], [3, 4]], names=["b", "a"])

    # 调用 join 方法，返回 join_idx 表示合并后的索引，lidx 和 ridx 是返回的索引器
    join_idx, lidx, ridx = midx1.join(midx2, return_indexers=True)

    # 期望的 ridx 结果是一个由 -1 组成的 NumPy 数组
    exp_ridx = np.array([-1, -1, -1, -1], dtype=np.intp)

    # 使用断言方法验证 join 后的索引 join_idx 与 midx1 是否相等
    tm.assert_index_equal(midx1, join_idx)
    # 断言 lidx 为 None
    assert lidx is None
    # 使用断言方法验证 ridx 与期望的 exp_ridx 是否相等
    tm.assert_numpy_array_equal(ridx, exp_ridx)


def test_join_multi_return_indexers():
    # 定义一个测试函数，用于验证 MultiIndex 对象的 join 方法在不返回索引器时的行为
    # GH 34074

    # 创建两个 MultiIndex 对象，一个有三个层级 ["a", "b", "c"]，另一个有两个层级 ["a", "b"]
    midx1 = MultiIndex.from_product([[1, 2], [3, 4], [5, 6]], names=["a", "b", "c"])
    midx2 = MultiIndex.from_product([[1, 2], [3, 4]], names=["a", "b"])

    # 调用 join 方法，不返回索引器
    result = midx1.join(midx2, return_indexers=False)
    
    # 使用断言方法验证 join 后的结果 result 是否与 midx1 相等
    tm.assert_index_equal(result, midx1)


def test_join_overlapping_interval_level():
    # 定义一个测试函数，用于验证 MultiIndex 对象的 join 方法在有重叠区间级别时的行为
    # GH 44096

    # 创建两个 MultiIndex 对象，分别表示包含区间的索引
    idx_1 = MultiIndex.from_tuples(
        [
            (1, Interval(0.0, 1.0)),
            (1, Interval(1.0, 2.0)),
            (1, Interval(2.0, 5.0)),
            (2, Interval(0.0, 1.0)),
            (2, Interval(1.0, 3.0)),  # 区间限制在 3.0 而非 2.0
            (2, Interval(3.0, 5.0)),
        ],
        names=["num", "interval"],
    )

    idx_2 = MultiIndex.from_tuples(
        [
            (1, Interval(2.0, 5.0)),
            (1, Interval(0.0, 1.0)),
            (1, Interval(1.0, 2.0)),
            (2, Interval(3.0, 5.0)),
            (2, Interval(0.0, 1.0)),
            (2, Interval(1.0, 3.0)),
        ],
        names=["num", "interval"],
    )

    # 创建期望的结果 MultiIndex 对象，表示在 outer 连接模式下的合并结果
    expected = MultiIndex.from_tuples(
        [
            (1, Interval(0.0, 1.0)),
            (1, Interval(1.0, 2.0)),
            (1, Interval(2.0, 5.0)),
            (2, Interval(0.0, 1.0)),
            (2, Interval(1.0, 3.0)),
            (2, Interval(3.0, 5.0)),
        ],
        names=["num", "interval"],
    )

    # 调用 join 方法，使用 outer 连接模式
    result = idx_1.join(idx_2, how="outer")

    # 使用断言方法验证 join 后的结果 result 是否与期望的 expected 相等
    tm.assert_index_equal(result, expected)


def test_join_midx_ea():
    # 定义一个测试函数，用于验证 MultiIndex 对象的 join 方法在包含 Int64 类型数据时的行为
    # GH#49277

    # 创建一个 MultiIndex 对象，包含两个层级，使用 Series 表示 Int64 类型数据
    midx = MultiIndex.from_arrays(
        [Series([1, 1, 3], dtype="Int64"), Series([1, 2, 3], dtype="Int64")],
        names=["a", "b"],
    )

    # 创建另一个 MultiIndex 对象，包含两个层级，其中一个层级使用 Series 表示 Int64 类型数据
    midx2 = MultiIndex.from_arrays(
        [Series([1], dtype="Int64"), Series([3], dtype="Int64")], names=["a", "c"]
    )

    # 调用 join 方法，使用 inner 连接模式
    result = midx.join(midx2, how="inner")

    # 创建期望的结果 MultiIndex 对象，包含三个层级
    expected = MultiIndex.from_arrays(
        [
            Series([1, 1], dtype="Int64"),
            Series([1, 2], dtype="Int64"),
            Series([3, 3], dtype="Int64"),
        ],
        names=["a", "b", "c"],
    )

    # 使用断言方法验证 join 后的结果 result 是否与期望的 expected 相等
    tm.assert_index_equal(result, expected)


def test_join_midx_string():
    # 定义一个测试函数，用于验证 MultiIndex 对象的 join 方法在包含 StringDtype 类型数据时的行为
    # GH#49277

    # 创建一个 MultiIndex 对象，包含两个层级，使用 Series 表示 StringDtype 类型数据
    midx = MultiIndex.from_arrays(
        [
            Series(["a", "a", "c"], dtype=StringDtype()),
            Series(["a", "b", "c"], dtype=StringDtype()),
        ],
        names=["a", "b"],
    )

    # 创建另一个 MultiIndex 对象，包含两个层级，其中一个层级使用 Series 表示 StringDtype 类型数据
    midx2 = MultiIndex.from_arrays(
        [Series(["a"], dtype=StringDtype()), Series(["c"], dtype=StringDtype())],
        names=["a", "c"],
    )

    # 调用 join 方法，使用 inner 连接模式
    result = midx.join(midx2, how="inner")
    # 创建一个 MultiIndex 对象，通过 from_arrays 方法，用于构建多层次索引
    expected = MultiIndex.from_arrays(
        # 使用三个 Series 对象来生成 MultiIndex 的各个层次的数据，指定数据类型为 StringDtype
        [
            Series(["a", "a"], dtype=StringDtype()),  # 第一层索引的数据
            Series(["a", "b"], dtype=StringDtype()),  # 第二层索引的数据
            Series(["c", "c"], dtype=StringDtype()),  # 第三层索引的数据
        ],
        # 指定各个层次的名称，依次对应于 a, b, c
        names=["a", "b", "c"],
    )
    
    # 使用 tm.assert_index_equal 进行断言，验证 result 和 expected 是否相等
    tm.assert_index_equal(result, expected)
# 定义一个测试函数，测试处理具有 NaN 值的多级索引的数据帧连接情况
def test_join_multi_with_nan():
    # GH29252: GitHub issue编号
    # 创建第一个数据帧，包含列"col1"，行索引为多级索引，其中id1为"A"，id2为1.0和2.0
    df1 = DataFrame(
        data={"col1": [1.1, 1.2]},
        index=MultiIndex.from_product([["A"], [1.0, 2.0]], names=["id1", "id2"]),
    )
    # 创建第二个数据帧，包含列"col2"，行索引为多级索引，其中id1为"A"，id2为NaN和2.0，使用np.nan表示NaN
    df2 = DataFrame(
        data={"col2": [2.1, 2.2]},
        index=MultiIndex.from_product([["A"], [np.nan, 2.0]], names=["id1", "id2"]),
    )
    # 将df1和df2按照索引连接起来
    result = df1.join(df2)
    # 创建预期的结果数据帧，包含列"col1"和"col2"，行索引与df1相同，对应位置有NaN值
    expected = DataFrame(
        data={"col1": [1.1, 1.2], "col2": [np.nan, 2.2]},
        index=MultiIndex.from_product([["A"], [1.0, 2.0]], names=["id1", "id2"]),
    )
    # 使用测试框架检查结果是否与预期相等
    tm.assert_frame_equal(result, expected)


# 使用参数化测试函数，测试具有不同数值的数据帧连接时的数据类型
@pytest.mark.parametrize("val", [0, 5])
def test_join_dtypes(any_numeric_ea_dtype, val):
    # GH#49830: GitHub issue编号
    # 创建一个多级索引midx，包含两个级别，第一级别为Series([1, 2])，第二级别为[3, 4]，数据类型由any_numeric_ea_dtype指定
    midx = MultiIndex.from_arrays([Series([1, 2], dtype=any_numeric_ea_dtype), [3, 4]])
    # 创建另一个多级索引midx2，包含两个级别，第一级别为Series([1, val, val])，第二级别为[3, 4, 4]，数据类型由any_numeric_ea_dtype指定
    midx2 = MultiIndex.from_arrays(
        [Series([1, val, val], dtype=any_numeric_ea_dtype), [3, 4, 4]]
    )
    # 使用outer方式将midx和midx2连接起来
    result = midx.join(midx2, how="outer")
    # 创建预期的多级索引，包含两个级别，第一级别为Series([val, val, 1, 2])，第二级别为[4, 4, 3, 4]，并按值排序
    expected = MultiIndex.from_arrays(
        [Series([val, val, 1, 2], dtype=any_numeric_ea_dtype), [4, 4, 3, 4]]
    ).sort_values()
    # 使用测试框架检查结果的索引是否与预期相等
    tm.assert_index_equal(result, expected)


# 测试处理所有值为NaN的情况下，数据帧连接的情况
def test_join_dtypes_all_nan(any_numeric_ea_dtype):
    # GH#49830: GitHub issue编号
    # 创建一个多级索引midx，包含两个级别，第一级别为Series([1, 2])，第二级别为[np.nan, np.nan]，数据类型由any_numeric_ea_dtype指定
    midx = MultiIndex.from_arrays(
        [Series([1, 2], dtype=any_numeric_ea_dtype), [np.nan, np.nan]]
    )
    # 创建另一个多级索引midx2，包含两个级别，第一级别为Series([1, 0, 0])，第二级别为[np.nan, np.nan, np.nan]，数据类型由any_numeric_ea_dtype指定
    midx2 = MultiIndex.from_arrays(
        [Series([1, 0, 0], dtype=any_numeric_ea_dtype), [np.nan, np.nan, np.nan]]
    )
    # 使用outer方式将midx和midx2连接起来
    result = midx.join(midx2, how="outer")
    # 创建预期的多级索引，包含两个级别，第一级别为Series([0, 0, 1, 2])，第二级别为[np.nan, np.nan, np.nan, np.nan]，所有值均为NaN
    expected = MultiIndex.from_arrays(
        [
            Series([0, 0, 1, 2], dtype=any_numeric_ea_dtype),
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )
    # 使用测试框架检查结果的索引是否与预期相等
    tm.assert_index_equal(result, expected)


# 测试连接具有不同索引级别的情况
def test_join_index_levels():
    # GH#53093: GitHub issue编号
    # 创建一个多级索引midx，包含两个元组，("a", "2019-02-01")和("a", "2019-02-01")
    midx = MultiIndex.from_tuples([("a", "2019-02-01"), ("a", "2019-02-01")])
    # 创建另一个多级索引midx2，包含一个元组("a", "2019-01-31")
    midx2 = MultiIndex.from_tuples([("a", "2019-01-31")])
    # 使用outer方式将midx和midx2连接起来
    result = midx.join(midx2, how="outer")
    # 创建预期的多级索引，包含三个元组，("a", "2019-01-31")，("a", "2019-02-01")和("a", "2019-02-01")
    expected = MultiIndex.from_tuples(
        [("a", "2019-01-31"), ("a", "2019-02-01"), ("a", "2019-02-01")]
    )
    # 使用测试框架检查结果的第二级别索引是否与预期相等
    tm.assert_index_equal(result.levels[1], expected.levels[1])
    # 使用测试框架检查结果的索引是否与预期相等
    tm.assert_index_equal(result, expected)
```