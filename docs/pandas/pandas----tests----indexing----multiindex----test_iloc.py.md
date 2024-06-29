# `D:\src\scipysrc\pandas\pandas\tests\indexing\multiindex\test_iloc.py`

```
# 导入必要的库
import numpy as np
import pytest

# 从 pandas 库中导入 DataFrame、MultiIndex、Series 类
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
)
# 导入 pandas 测试工具模块
import pandas._testing as tm

# 定义一个 pytest 的 fixture，用于创建一个简单的多级索引 DataFrame
@pytest.fixture
def simple_multiindex_dataframe():
    """
    Factory function to create simple 3 x 3 dataframe with
    both columns and row MultiIndex using supplied data or
    random data by default.
    """
    # 使用随机生成的正态分布数据创建一个 3 x 3 的 DataFrame
    data = np.random.default_rng(2).standard_normal((3, 3))
    return DataFrame(
        data, columns=[[2, 2, 4], [6, 8, 10]], index=[[4, 4, 8], [8, 10, 12]]
    )

# 定义一个参数化测试，用于测试 iloc 方法返回的 Series 对象
@pytest.mark.parametrize(
    "indexer, expected",
    [
        (
            lambda df: df.iloc[0],
            lambda arr: Series(arr[0], index=[[2, 2, 4], [6, 8, 10]], name=(4, 8)),
        ),
        (
            lambda df: df.iloc[2],
            lambda arr: Series(arr[2], index=[[2, 2, 4], [6, 8, 10]], name=(8, 12)),
        ),
        (
            lambda df: df.iloc[:, 2],
            lambda arr: Series(arr[:, 2], index=[[4, 4, 8], [8, 10, 12]], name=(4, 10)),
        ),
    ],
)
def test_iloc_returns_series(indexer, expected, simple_multiindex_dataframe):
    # 准备测试数据
    df = simple_multiindex_dataframe
    arr = df.values
    # 执行索引操作
    result = indexer(df)
    # 准备预期结果
    expected = expected(arr)
    # 断言结果与预期结果相等
    tm.assert_series_equal(result, expected)

# 测试 iloc 方法返回的 DataFrame 对象
def test_iloc_returns_dataframe(simple_multiindex_dataframe):
    # 准备测试数据
    df = simple_multiindex_dataframe
    # 执行索引操作
    result = df.iloc[[0, 1]]
    # 准备预期结果
    expected = df.xs(4, drop_level=False)
    # 断言结果与预期结果相等
    tm.assert_frame_equal(result, expected)

# 测试 iloc 方法返回的标量值
def test_iloc_returns_scalar(simple_multiindex_dataframe):
    # 准备测试数据
    df = simple_multiindex_dataframe
    arr = df.values
    # 执行索引操作
    result = df.iloc[2, 2]
    # 准备预期结果
    expected = arr[2, 2]
    # 断言结果与预期结果相等
    assert result == expected

# 测试 iloc 方法进行多项选择时返回的 DataFrame 对象
def test_iloc_getitem_multiple_items():
    # 创建一个具有多级索引的 DataFrame
    tup = zip(*[["a", "a", "b", "b"], ["x", "y", "x", "y"]])
    index = MultiIndex.from_tuples(tup)
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=index)
    # 执行索引操作
    result = df.iloc[[2, 3]]
    # 准备预期结果
    expected = df.xs("b", drop_level=False)
    # 断言结果与预期结果相等
    tm.assert_frame_equal(result, expected)

# 测试 iloc 方法使用标签进行索引时的行为
def test_iloc_getitem_labels():
    # 创建一个具有多级索引的 DataFrame
    arr = np.random.default_rng(2).standard_normal((4, 3))
    df = DataFrame(
        arr,
        columns=[["i", "i", "j"], ["A", "A", "B"]],
        index=[["i", "i", "j", "k"], ["X", "X", "Y", "Y"]],
    )
    # 执行索引操作
    result = df.iloc[2, 2]
    # 准备预期结果
    expected = arr[2, 2]
    # 断言结果与预期结果相等
    assert result == expected

# 测试 DataFrame 使用 iloc 进行切片操作
def test_frame_getitem_slice(multiindex_dataframe_random_data):
    # 准备测试数据
    df = multiindex_dataframe_random_data
    # 执行切片操作
    result = df.iloc[:4]
    # 准备预期结果
    expected = df[:4]
    # 断言结果与预期结果相等
    tm.assert_frame_equal(result, expected)

# 测试 DataFrame 使用 iloc 进行切片赋值操作
def test_frame_setitem_slice(multiindex_dataframe_random_data):
    # 准备测试数据
    df = multiindex_dataframe_random_data
    # 执行切片赋值操作
    df.iloc[:4] = 0
    # 断言前四行的值全部为 0
    assert (df.values[:4] == 0).all()
    # 断言后四行的值不全为 0
    assert (df.values[4:] != 0).all()

# 测试处理索引歧义的 bug 1678
def test_indexing_ambiguity_bug_1678():
    # 创建一个具有多级列索引的 MultiIndex 对象
    columns = MultiIndex.from_tuples(
        [("Ohio", "Green"), ("Ohio", "Red"), ("Colorado", "Green")]
    )
    # 使用 MultiIndex.from_tuples 创建一个多级索引对象，包含四个元组作为索引的标签
    index = MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])
    
    # 使用 DataFrame 创建一个数据框对象，其中数据由 np.arange(12) 生成并重塑为 4x3 的形状，
    # 索引使用上面创建的多级索引对象 index，列名由变量 columns 指定
    df = DataFrame(np.arange(12).reshape((4, 3)), index=index, columns=columns)
    
    # 从数据框 df 中选择所有行和第二列，结果为一个 Series 对象 result
    result = df.iloc[:, 1]
    
    # 从数据框 df 中选择所有行和列标签为 ("Ohio", "Red") 的数据，结果为一个 Series 对象 expected
    expected = df.loc[:, ("Ohio", "Red")]
    
    # 使用 tm.assert_series_equal 函数比较 result 和 expected 两个 Series 对象是否相等，
    # 如果不相等，则抛出 AssertionError
    tm.assert_series_equal(result, expected)
# 定义测试函数，用于测试 DataFrame 的 iloc 功能
def test_iloc_integer_locations():
    # GH 13797，引用 GitHub 问题编号，标识测试背景
    data = [
        ["str00", "str01"],
        ["str10", "str11"],
        ["str20", "srt21"],  # 错误：应为 "str21"，修正拼写错误
        ["str30", "str31"],
        ["str40", "str41"],
    ]

    # 创建 MultiIndex 对象，作为 DataFrame 的索引
    index = MultiIndex.from_tuples(
        [("CC", "A"), ("CC", "B"), ("CC", "B"), ("BB", "a"), ("BB", "b")]
    )

    # 期望的 DataFrame，使用给定的数据和索引
    expected = DataFrame(data)
    df = DataFrame(data, index=index)

    # 创建结果 DataFrame，使用嵌套列表推导式来访问 df 的元素并构建新的 DataFrame
    result = DataFrame([[df.iloc[r, c] for c in range(2)] for r in range(5)])

    # 使用 pytest 的 assert_frame_equal 断言函数比较结果 DataFrame 和期望的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest 的参数化装饰器，定义多个测试用例来测试 iloc_setitem_int_multiindex_series 函数
@pytest.mark.parametrize(
    "data, indexes, values, expected_k",
    [
        # 测试：在 MultiIndex 的第一个级别中没有索引器值的情况
        ([[2, 22, 5], [2, 33, 6]], [0, -1, 1], [2, 3, 1], [7, 10]),
        # 测试：类似于问题中示例 1 的情况
        ([[1, 22, 555], [1, 33, 666]], [0, -1, 1], [200, 300, 100], [755, 1066]),
        # 测试：类似于问题中示例 2 的情况
        ([[1, 3, 7], [2, 4, 8]], [0, -1, 1], [10, 10, 1000], [17, 1018]),
        # 测试：类似于问题中示例 3 的情况
        ([[1, 11, 4], [2, 22, 5], [3, 33, 6]], [0, -1, 1], [4, 7, 10], [8, 15, 13]),
    ],
)
def test_iloc_setitem_int_multiindex_series(data, indexes, values, expected_k):
    # GH17148，引用 GitHub 问题编号，标识测试背景
    # 创建 DataFrame，使用给定的数据和列名
    df = DataFrame(data=data, columns=["i", "j", "k"])
    # 将 "i" 和 "j" 列设置为索引，创建 MultiIndex
    df = df.set_index(["i", "j"])

    # 复制 "k" 列作为 Series 对象
    series = df.k.copy()
    # 使用 for 循环和 zip 函数，修改 Series 对象的特定位置的值
    for i, v in zip(indexes, values):
        series.iloc[i] += v

    # 将 DataFrame 的 "k" 列更新为预期值列表 expected_k
    df["k"] = expected_k
    # 期望的结果是 DataFrame 的 "k" 列
    expected = df.k
    # 使用 pytest 的 assert_series_equal 断言函数比较 Series 对象是否相等
    tm.assert_series_equal(series, expected)


# 定义测试函数，测试 DataFrame 的 iloc 和 xs 方法
def test_getitem_iloc(multiindex_dataframe_random_data):
    # 获取随机生成的多级索引 DataFrame
    df = multiindex_dataframe_random_data
    # 使用 iloc 方法获取第 2 行数据，返回 Series 对象
    result = df.iloc[2]
    # 使用 xs 方法获取索引为 df.index[2] 的数据，返回期望的 Series 对象
    expected = df.xs(df.index[2])
    # 使用 pytest 的 assert_series_equal 断言函数比较两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
```