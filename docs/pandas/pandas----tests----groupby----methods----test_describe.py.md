# `D:\src\scipysrc\pandas\pandas\tests\groupby\methods\test_describe.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库
from pandas import (  # 从 Pandas 库中导入多个子模块和类
    DataFrame,  # 用于操作和处理数据的二维表格
    Index,  # Pandas 的索引对象
    MultiIndex,  # Pandas 多级索引对象
    Series,  # 用于操作和处理数据的一维标记数组
    Timestamp,  # Pandas 的时间戳对象
    date_range,  # 生成时间序列的函数
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具模块


def test_apply_describe_bug(multiindex_dataframe_random_data):
    grouped = multiindex_dataframe_random_data.groupby(level="first")
    grouped.describe()  # 对分组数据执行描述统计操作


def test_series_describe_multikey():
    ts = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    grouped = ts.groupby([lambda x: x.year, lambda x: x.month])
    result = grouped.describe()
    tm.assert_series_equal(result["mean"], grouped.mean(), check_names=False)
    tm.assert_series_equal(result["std"], grouped.std(), check_names=False)
    tm.assert_series_equal(result["min"], grouped.min(), check_names=False)


def test_series_describe_single():
    ts = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    grouped = ts.groupby(lambda x: x.month)
    result = grouped.apply(lambda x: x.describe())
    expected = grouped.describe().stack()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("keys", ["key1", ["key1", "key2"]])
def test_series_describe_as_index(as_index, keys):
    # GH#49256
    df = DataFrame(
        {
            "key1": ["one", "two", "two", "three", "two"],
            "key2": ["one", "two", "two", "three", "two"],
            "foo2": [1, 2, 4, 4, 6],
        }
    )
    gb = df.groupby(keys, as_index=as_index)["foo2"]
    result = gb.describe()
    expected = DataFrame(
        {
            "key1": ["one", "three", "two"],
            "count": [1.0, 1.0, 3.0],
            "mean": [1.0, 4.0, 4.0],
            "std": [np.nan, np.nan, 2.0],
            "min": [1.0, 4.0, 2.0],
            "25%": [1.0, 4.0, 3.0],
            "50%": [1.0, 4.0, 4.0],
            "75%": [1.0, 4.0, 5.0],
            "max": [1.0, 4.0, 6.0],
        }
    )
    if len(keys) == 2:
        expected.insert(1, "key2", expected["key1"])
    if as_index:
        expected = expected.set_index(keys)
    tm.assert_frame_equal(result, expected)


def test_frame_describe_multikey(tsframe):
    grouped = tsframe.groupby([lambda x: x.year, lambda x: x.month])
    result = grouped.describe()
    desc_groups = []
    for col in tsframe:
        group = grouped[col].describe()
        # GH 17464 - Remove duplicate MultiIndex levels
        group_col = MultiIndex(
            levels=[[col], group.columns],
            codes=[[0] * len(group.columns), range(len(group.columns))],
        )
        group = DataFrame(group.values, columns=group_col, index=group.index)
        desc_groups.append(group)
    expected = pd.concat(desc_groups, axis=1)
    tm.assert_frame_equal(result, expected)


def test_frame_describe_tupleindex():
    # GH 14848 - regression from 0.19.0 to 0.19.1
    name = "k"
    # 创建一个 DataFrame 对象，包含两列数据："x" 和给定的 name 列
    df = DataFrame(
        {
            "x": [1, 2, 3, 4, 5] * 3,  # 列 'x' 包含重复的整数序列
            name: [(0, 0, 1), (0, 1, 0), (1, 0, 0)] * 5,  # 给定的 name 列包含元组的重复序列
        }
    )
    # 对 DataFrame 按照 name 列进行分组，并计算描述统计信息
    result = df.groupby(name).describe()
    # 创建一个预期的 DataFrame，包含预定义的统计值序列，行索引为元组的列表，列索引为多级索引
    expected = DataFrame(
        [[5.0, 3.0, 1.581139, 1.0, 2.0, 3.0, 4.0, 5.0]] * 3,
        index=Index([(0, 0, 1), (0, 1, 0), (1, 0, 0)], tupleize_cols=False, name=name),
        columns=MultiIndex.from_arrays(
            [["x"] * 8, ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        ),
    )
    # 使用测试工具来比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
def test_frame_describe_unstacked_format():
    # GH 4792
    # 创建一个价格字典，每个时间戳对应一个价格
    prices = {
        Timestamp("2011-01-06 10:59:05", tz=None): 24990,
        Timestamp("2011-01-06 12:43:33", tz=None): 25499,
        Timestamp("2011-01-06 12:54:09", tz=None): 25499,
    }
    # 创建一个交易量字典，每个时间戳对应一个交易量
    volumes = {
        Timestamp("2011-01-06 10:59:05", tz=None): 1500000000,
        Timestamp("2011-01-06 12:43:33", tz=None): 5000000000,
        Timestamp("2011-01-06 12:54:09", tz=None): 100000000,
    }
    # 从价格和交易量字典创建一个数据帧
    df = DataFrame({"PRICE": prices, "VOLUME": volumes})
    # 对价格进行分组，并对每组的交易量进行描述统计
    result = df.groupby("PRICE").VOLUME.describe()
    # 创建一个预期的数据帧，包含每个价格的描述统计数据
    data = [
        df[df.PRICE == 24990].VOLUME.describe().values.tolist(),
        df[df.PRICE == 25499].VOLUME.describe().values.tolist(),
    ]
    expected = DataFrame(
        data,
        index=Index([24990, 25499], name="PRICE"),
        columns=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
    )
    # 使用测试框架的函数验证结果与预期是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:"
    "indexing past lexsort depth may impact performance:"
    "pandas.errors.PerformanceWarning"
)
@pytest.mark.parametrize("keys", [["a1"], ["a1", "a2"]])
def test_describe_with_duplicate_output_column_names(as_index, keys):
    # GH 35314
    # 创建一个数据帧，包含重复的输出列名和指定的键
    df = DataFrame(
        {
            "a1": [99, 99, 99, 88, 88, 88],
            "a2": [99, 99, 99, 88, 88, 88],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [10, 20, 30, 40, 50, 60],
        },
        columns=["a1", "a2", "b", "b"],  # 指定列名
        copy=False,
    )
    # 如果键是["a1"]，则删除数据帧中的"a2"列
    if keys == ["a1"]:
        df = df.drop(columns="a2")

    # 创建一个预期的数据帧，包含描述统计数据，具体内容根据键的长度来确定
    expected = (
        DataFrame.from_records(
            [
                ("b", "count", 3.0, 3.0),
                ("b", "mean", 5.0, 2.0),
                ("b", "std", 1.0, 1.0),
                ("b", "min", 4.0, 1.0),
                ("b", "25%", 4.5, 1.5),
                ("b", "50%", 5.0, 2.0),
                ("b", "75%", 5.5, 2.5),
                ("b", "max", 6.0, 3.0),
                ("b", "count", 3.0, 3.0),
                ("b", "mean", 5.0, 2.0),
                ("b", "std", 1.0, 1.0),
                ("b", "min", 4.0, 1.0),
                ("b", "25%", 4.5, 1.5),
                ("b", "50%", 5.0, 2.0),
                ("b", "75%", 5.5, 2.5),
                ("b", "max", 6.0, 3.0),
            ],
        )
        .set_index([0, 1])
        .T
    )
    expected.columns.names = [None, None]
    # 如果键的长度为2，创建一个多级索引
    if len(keys) == 2:
        expected.index = MultiIndex(
            levels=[[88, 99], [88, 99]], codes=[[0, 1], [0, 1]], names=["a1", "a2"]
        )
    else:
        expected.index = Index([88, 99], name="a1")

    # 如果不是作为索引(as_index)，则重置预期的数据帧索引
    if not as_index:
        expected = expected.reset_index()

    # 使用测试框架的函数验证结果与预期是否相等
    result = df.groupby(keys, as_index=as_index).describe()
    tm.assert_frame_equal(result, expected)


def test_describe_duplicate_columns():
    # GH#50806
    # 创建一个数据帧，包含重复的列名
    df = DataFrame([[0, 1, 2, 3]])
    # 将列名设置为[0, 1, 2, 0]
    df.columns = [0, 1, 2, 0]
    # 根据 df[1] 列进行分组
    gb = df.groupby(df[1])
    # 对分组后的数据进行描述统计，不包含百分位数
    result = gb.describe(percentiles=[])
    # 定义一个包含列名的列表
    columns = ["count", "mean", "std", "min", "50%", "max"]
    # 创建包含多个数据框的列表，每个数据框包含一行数据，每行数据中 'count' 列为 1.0，其余列根据 val 变化
    frames = [
        DataFrame([[1.0, val, np.nan, val, val, val]], index=[1], columns=columns)
        for val in (0.0, 2.0, 3.0)
    ]
    # 使用 pd.concat 将 frames 列表中的数据框按列拼接成一个数据框，axis=1 表示按列拼接
    expected = pd.concat(frames, axis=1)
    # 设置 expected 数据框的列名为 MultiIndex，levels 是一个包含两个层级的列表，codes 指定了每个列的层级编码
    expected.columns = MultiIndex(
        levels=[[0, 2], columns],  # 两个层级：第一层是 [0, 2]，第二层是 columns 中定义的列名
        codes=[6 * [0] + 6 * [1] + 6 * [0], 3 * list(range(6))]  # 指定每列对应的层级编码
    )
    # 设置 expected 数据框的索引名为 [1]
    expected.index.names = [1]
    # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
# 定义测试函数，用于描述非 Cython 路径
def test_describe_non_cython_paths():
    # GH#5610: 非 Cython 调用不应包括 grouper
    # 创建包含混合数据的 DataFrame 对象
    df = DataFrame(
        [[1, 2, "foo"], [1, np.nan, "bar"], [3, np.nan, "baz"]],
        columns=["A", "B", "C"],
    )
    # 根据列"A"进行分组
    gb = df.groupby("A")
    # 期望的索引对象
    expected_index = Index([1, 3], name="A")
    # 期望的多级列对象
    expected_col = MultiIndex(
        levels=[["B"], ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]],
        codes=[[0] * 8, list(range(8))],
    )
    # 创建期望的 DataFrame 对象
    expected = DataFrame(
        [
            [1.0, 2.0, np.nan, 2.0, 2.0, 2.0, 2.0, 2.0],
            [0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
        index=expected_index,
        columns=expected_col,
    )
    # 执行描述操作，获取结果 DataFrame 对象
    result = gb.describe()
    # 检查结果与期望是否一致
    tm.assert_frame_equal(result, expected)

    # 创建不含索引的分组对象
    gni = df.groupby("A", as_index=False)
    # 重置期望 DataFrame 的索引
    expected = expected.reset_index()
    # 执行描述操作，获取结果 DataFrame 对象
    result = gni.describe()
    # 检查结果与期望是否一致
    tm.assert_frame_equal(result, expected)


# 使用参数化装饰器指定数据类型和关键字参数
@pytest.mark.parametrize("dtype", [int, float, object])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"percentiles": [0.10, 0.20, 0.30], "include": "all", "exclude": None},
        {"percentiles": [0.10, 0.20, 0.30], "include": None, "exclude": ["int"]},
        {"percentiles": [0.10, 0.20, 0.30], "include": ["int"], "exclude": None},
    ],
)
# 定义测试空数据集的分组操作
def test_groupby_empty_dataset(dtype, kwargs):
    # GH#41575: 空数据集的分组描述
    # 创建包含一行数据的 DataFrame 对象
    df = DataFrame([[1, 2, 3]], columns=["A", "B", "C"], dtype=dtype)
    # 将列"B"转换为整数类型
    df["B"] = df["B"].astype(int)
    # 将列"C"转换为浮点数类型
    df["C"] = df["C"].astype(float)

    # 执行空数据集的分组描述操作，获取结果 DataFrame 对象
    result = df.iloc[:0].groupby("A").describe(**kwargs)
    # 根据期望的描述结果重置索引并截取空行
    expected = df.groupby("A").describe(**kwargs).reset_index(drop=True).iloc[:0]
    # 检查结果与期望是否一致
    tm.assert_frame_equal(result, expected)

    # 执行空数据集的分组描述操作，获取结果 Series 对象
    result = df.iloc[:0].groupby("A").B.describe(**kwargs)
    # 根据期望的描述结果重置索引并截取空行
    expected = df.groupby("A").B.describe(**kwargs).reset_index(drop=True).iloc[:0]
    # 设置期望结果的索引为空 Index 对象
    expected.index = Index([])
    # 检查结果与期望是否一致
    tm.assert_frame_equal(result, expected)
```