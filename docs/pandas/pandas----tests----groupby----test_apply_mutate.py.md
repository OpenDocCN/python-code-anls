# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_apply_mutate.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
import pandas._testing as tm  # 导入 Pandas 内部测试模块

def test_group_by_copy():
    # GH#44803
    # 创建一个 DataFrame 对象 df，包含姓名和年龄，将姓名设置为索引
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carl"],
            "age": [20, 21, 20],
        }
    ).set_index("name")

    # 设置警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用 assert_produces_warning 检查是否产生 DeprecationWarning，并匹配消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按年龄分组，不保留组键，应用 lambda 函数返回组
        grp_by_same_value = df.groupby(["age"], group_keys=False).apply(
            lambda group: group
        )
    # 设置警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用 assert_produces_warning 检查是否产生 DeprecationWarning，并匹配消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按年龄分组，不保留组键，应用 lambda 函数返回组的副本
        grp_by_copy = df.groupby(["age"], group_keys=False).apply(
            lambda group: group.copy()
        )
    # 断言 grp_by_same_value 和 grp_by_copy 应该相等
    tm.assert_frame_equal(grp_by_same_value, grp_by_copy)


def test_mutate_groups():
    # GH3380
    # 创建一个 DataFrame df 包含 cat1, cat2, cat3 和 val 列
    df = pd.DataFrame(
        {
            "cat1": ["a"] * 8 + ["b"] * 6,
            "cat2": ["c"] * 2 + ["d"] * 2 + ["e"] * 2 + ["f"] * 2 + ["c"] * 2 + ["d"] * 2 + ["e"] * 2,
            "cat3": [f"g{x}" for x in range(1, 15)],
            "val": np.random.default_rng(2).integers(100, size=14),
        }
    )

    # 定义函数 f_copy，复制组数据，添加 rank 列，并按 cat2 分组取最小值
    def f_copy(x):
        x = x.copy()  # 复制 DataFrame x
        x["rank"] = x.val.rank(method="min")  # 添加 rank 列，根据 val 列计算排名
        return x.groupby("cat2")["rank"].min()  # 按 cat2 分组，返回 rank 列的最小值

    # 定义函数 f_no_copy，直接在组数据上添加 rank 列，并按 cat2 分组取最小值
    def f_no_copy(x):
        x["rank"] = x.val.rank(method="min")  # 添加 rank 列，根据 val 列计算排名
        return x.groupby("cat2")["rank"].min()  # 按 cat2 分组，返回 rank 列的最小值

    # 设置警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用 assert_produces_warning 检查是否产生 DeprecationWarning，并匹配消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按 cat1 分组应用 f_copy 函数
        grpby_copy = df.groupby("cat1").apply(f_copy)
    # 设置警告消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按 cat1 分组应用 f_no_copy 函数
        grpby_no_copy = df.groupby("cat1").apply(f_no_copy)
    # 断言 grpby_copy 和 grpby_no_copy 应该相等
    tm.assert_series_equal(grpby_copy, grpby_no_copy)


def test_no_mutate_but_looks_like():
    # GH 8467
    # 创建一个 DataFrame 包含 key 和 value 列
    df = pd.DataFrame({"key": [1, 1, 1, 2, 2, 2, 3, 3, 3], "value": range(9)})

    # 设置警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用 assert_produces_warning 检查是否产生 DeprecationWarning，并匹配消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按 key 分组，组键保留，应用 lambda 函数返回 x[:].key 列
        result1 = df.groupby("key", group_keys=True).apply(lambda x: x[:].key)
    # 设置警告消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按 key 分组，组键保留，应用 lambda 函数返回 x.key 列
        result2 = df.groupby("key", group_keys=True).apply(lambda x: x.key)
    # 断言 result1 和 result2 应该相等
    tm.assert_series_equal(result1, result2)


def test_apply_function_with_indexing():
    # GH: 33058
    # 创建一个 DataFrame 包含 col1 和 col2 列
    df = pd.DataFrame(
        {"col1": ["A", "A", "A", "B", "B", "B"], "col2": [1, 2, 3, 4, 5, 6]}
    )

    # 定义函数 fn，在组内修改最后一行的 col2 值为 0，并返回 col2 列
    def fn(x):
        x.loc[x.index[-1], "col2"] = 0  # 将组内最后一行的 col2 列设为 0
        return x.col2  # 返回 col2 列

    # 设置警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用 assert_produces_warning 检查是否产生 DeprecationWarning，并匹配消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按 col1 分组，不保留组键，应用 fn 函数
        result = df.groupby(["col1"], as_index=False).apply(fn)
    # 创建一个 Pandas Series 对象，其中包含指定的数据列表 [1, 2, 0, 4, 5, 0]
    # 设置 Series 的索引为 range(6)，即 [0, 1, 2, 3, 4, 5]
    # 设置 Series 的名称为 "col2"
    expected = pd.Series(
        [1, 2, 0, 4, 5, 0],
        index=range(6),
        name="col2",
    )
    
    # 使用 Pandas Testing 模块（tm）中的 assert_series_equal 函数比较 result 和 expected 两个 Series 对象
    tm.assert_series_equal(result, expected)
```