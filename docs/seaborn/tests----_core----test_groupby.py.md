# `D:\src\scipysrc\seaborn\tests\_core\test_groupby.py`

```
# 导入必要的库：numpy和pandas用于数据处理，pytest用于单元测试，assert_array_equal用于断言数组相等性
import numpy as np
import pandas as pd

import pytest  # 导入pytest库进行单元测试
from numpy.testing import assert_array_equal  # 导入numpy的数组相等性断言函数

from seaborn._core.groupby import GroupBy  # 从seaborn库中导入GroupBy类


@pytest.fixture
def df():
    # 创建一个测试用的DataFrame，包含列名为["a", "b", "x", "y"]的数据
    return pd.DataFrame(
        columns=["a", "b", "x", "y"],
        data=[
            ["a", "g", 1, .2],
            ["b", "h", 3, .5],
            ["a", "f", 2, .8],
            ["a", "h", 1, .3],
            ["b", "f", 2, .4],
        ]
    )


def test_init_from_list():
    # 测试以列表形式初始化GroupBy对象
    g = GroupBy(["a", "c", "b"])
    # 断言初始化后的顺序与预期一致
    assert g.order == {"a": None, "c": None, "b": None}


def test_init_from_dict():
    # 测试以字典形式初始化GroupBy对象
    order = {"a": [3, 2, 1], "c": None, "b": ["x", "y", "z"]}
    g = GroupBy(order)
    # 断言初始化后的顺序与预期一致
    assert g.order == order


def test_init_requires_order():
    # 测试初始化GroupBy对象时需要至少一个排序变量，否则应该抛出 ValueError 异常
    with pytest.raises(ValueError, match="GroupBy requires at least one"):
        GroupBy([])


def test_at_least_one_grouping_variable_required(df):
    # 测试在GroupBy对象中至少需要一个分组变量，否则应该抛出 ValueError 异常
    with pytest.raises(ValueError, match="No grouping variables are present"):
        GroupBy(["z"]).agg(df, x="mean")


def test_agg_one_grouper(df):
    # 测试在一个分组变量下进行聚合操作
    res = GroupBy(["a"]).agg(df, {"y": "max"})
    # 断言聚合结果的索引和列名符合预期
    assert_array_equal(res.index, [0, 1])
    assert_array_equal(res.columns, ["a", "y"])
    assert_array_equal(res["a"], ["a", "b"])
    assert_array_equal(res["y"], [.8, .5])


def test_agg_two_groupers(df):
    # 测试在两个分组变量下进行聚合操作
    res = GroupBy(["a", "x"]).agg(df, {"y": "min"})
    # 断言聚合结果的索引和列名符合预期
    assert_array_equal(res.index, [0, 1, 2, 3, 4, 5])
    assert_array_equal(res.columns, ["a", "x", "y"])
    assert_array_equal(res["a"], ["a", "a", "a", "b", "b", "b"])
    assert_array_equal(res["x"], [1, 2, 3, 1, 2, 3])
    assert_array_equal(res["y"], [.2, .8, np.nan, np.nan, .4, .5])


def test_agg_two_groupers_ordered(df):
    # 测试在两个分组变量下进行聚合操作，并且结果按照指定顺序排列
    order = {"b": ["h", "g", "f"], "x": [3, 2, 1]}
    res = GroupBy(order).agg(df, {"a": "min", "y": lambda x: x.iloc[0]})
    # 断言聚合结果的索引和列名符合预期
    assert_array_equal(res.index, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert_array_equal(res.columns, ["a", "b", "x", "y"])
    assert_array_equal(res["b"], ["h", "h", "h", "g", "g", "g", "f", "f", "f"])
    assert_array_equal(res["x"], [3, 2, 1, 3, 2, 1, 3, 2, 1])
    # 断言a列中的缺失值和非缺失值符合预期
    T, F = True, False
    assert_array_equal(res["a"].isna(), [F, T, F, T, T, F, T, F, T])
    assert_array_equal(res["a"].dropna(), ["b", "a", "a", "a"])
    assert_array_equal(res["y"].dropna(), [.5, .3, .2, .8])


def test_apply_no_grouper(df):
    # 测试在没有分组变量的情况下应用函数操作
    df = df[["x", "y"]]
    res = GroupBy(["a"]).apply(df, lambda x: x.sort_values("x"))
    # 断言应用函数后的结果列名符合预期
    assert_array_equal(res.columns, ["x", "y"])
    assert_array_equal(res["x"], df["x"].sort_values())
    assert_array_equal(res["y"], df.loc[np.argsort(df["x"]), "y"])


def test_apply_one_grouper(df):
    # 测试在一个分组变量的情况下应用函数操作
    res = GroupBy(["a"]).apply(df, lambda x: x.sort_values("x"))
    # 断言应用函数后的结果索引和列名符合预期
    assert_array_equal(res.index, [0, 1, 2, 3, 4])
    assert_array_equal(res.columns, ["a", "b", "x", "y"])
    assert_array_equal(res["a"], ["a", "a", "a", "b", "b"])
    assert_array_equal(res["b"], ["g", "h", "f", "f", "h"])
    assert_array_equal(res["x"], [1, 1, 2, 2, 3])


def test_apply_mutate_columns(df):
    # 这个函数测试应用函数时修改列名的操作，代码未完整给出，需要继续完成
    pass
    # 创建一个包含 0 到 4 的 NumPy 数组
    xx = np.arange(0, 5)
    # 初始化一个空列表用于存储拟合结果
    hats = []

    # 定义一个函数 polyfit，用于对数据框 df 中的 x 和 y 列进行一次多项式拟合
    def polyfit(df):
        # 使用 np.polyfit 对 df 的 "x" 列和 "y" 列进行一次多项式拟合，得到拟合系数
        fit = np.polyfit(df["x"], df["y"], 1)
        # 使用 np.polyval 对拟合系数 fit 和全局数组 xx 进行多项式求值，得到拟合结果 hat
        hat = np.polyval(fit, xx)
        # 将拟合结果 hat 添加到全局列表 hats 中
        hats.append(hat)
        # 返回一个包含 xx 和 hat 列的新数据框
        return pd.DataFrame(dict(x=xx, y=hat))

    # 对数据框 df 按 "a" 列进行分组，然后对每组应用 polyfit 函数
    res = GroupBy(["a"]).apply(df, polyfit)

    # 断言结果的索引应与 xx 大小的两倍相等
    assert_array_equal(res.index, np.arange(xx.size * 2))
    # 断言结果的列应为 ["a", "x", "y"]
    assert_array_equal(res.columns, ["a", "x", "y"])
    # 断言结果的 "a" 列应为 ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]
    assert_array_equal(res["a"], ["a"] * xx.size + ["b"] * xx.size)
    # 断言结果的 "x" 列应为 xx 列表的两倍
    assert_array_equal(res["x"], xx.tolist() + xx.tolist())
    # 断言结果的 "y" 列应为 hats 列表的连接结果
    assert_array_equal(res["y"], np.concatenate(hats))
# 定义一个函数 test_apply_replace_columns，用于测试应用替换列操作
def test_apply_replace_columns(df):

    # 定义一个内部函数 add_sorted_cumsum，用于对传入的 DataFrame 进行排序累加求和操作
    def add_sorted_cumsum(df):

        # 获取 DataFrame 列 "x" 的排序值
        x = df["x"].sort_values()
        # 使用排序后的 "x" 列索引，计算 "y" 列的累积和
        z = df.loc[x.index, "y"].cumsum()
        # 返回一个新的 DataFrame，包含 "x" 列排序后的值和累积和 "z" 列的值
        return pd.DataFrame(dict(x=x.values, z=z.values))

    # 使用 GroupBy 对象调用 apply 方法，将 add_sorted_cumsum 函数应用于 df 数据框
    res = GroupBy(["a"]).apply(df, add_sorted_cumsum)
    
    # 断言结果 DataFrame 的索引与原始 DataFrame 的索引相等
    assert_array_equal(res.index, df.index)
    # 断言结果 DataFrame 的列名与预期的列名 ["a", "x", "z"] 相等
    assert_array_equal(res.columns, ["a", "x", "z"])
    # 断言结果 DataFrame 中 "a" 列的值与预期的值数组相等
    assert_array_equal(res["a"], ["a", "a", "a", "b", "b"])
    # 断言结果 DataFrame 中 "x" 列的值与预期的值数组相等
    assert_array_equal(res["x"], [1, 1, 2, 2, 3])
    # 断言结果 DataFrame 中 "z" 列的值与预期的值数组相等
    assert_array_equal(res["z"], [.2, .5, 1.3, .4, .9])
```