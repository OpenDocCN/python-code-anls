# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_unstack.py`

```
# 导入所需的库
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm


# 定义测试函数：测试 unstack 方法是否正确保留对象类型
def test_unstack_preserves_object():
    # 创建一个多级索引对象 mi
    mi = MultiIndex.from_product([["bar", "foo"], ["one", "two"]])
    
    # 创建一个 Series 对象 ser，其中包含对象类型的数据
    ser = Series(np.arange(4.0), index=mi, dtype=object)
    
    # 对 ser 应用 unstack 方法，得到结果 res1，并断言其所有数据类型仍为对象类型
    res1 = ser.unstack()
    assert (res1.dtypes == object).all()
    
    # 对 ser 应用带有 level 参数的 unstack 方法，得到结果 res2，并断言其所有数据类型仍为对象类型
    res2 = ser.unstack(level=0)
    assert (res2.dtypes == object).all()


# 定义测试函数：测试 unstack 方法的功能
def test_unstack():
    # 创建一个多级索引对象 index
    index = MultiIndex(
        levels=[["bar", "foo"], ["one", "three", "two"]],
        codes=[[1, 1, 0, 0], [0, 1, 0, 2]],
    )
    
    # 创建一个 Series 对象 s，包含索引为 index 的数据
    s = Series(np.arange(4.0), index=index)
    
    # 对 s 应用 unstack 方法，将其展开为 DataFrame 对象 unstacked
    unstacked = s.unstack()
    
    # 创建预期的 DataFrame 对象 expected
    expected = DataFrame(
        [[2.0, np.nan, 3.0], [0.0, 1.0, np.nan]],
        index=["bar", "foo"],
        columns=["one", "three", "two"],
    )
    
    # 使用 pandas._testing 模块的 assert_frame_equal 方法，断言 unstacked 和 expected 相等
    tm.assert_frame_equal(unstacked, expected)
    
    # 对 s 应用带有 level 参数的 unstack 方法，得到结果 unstacked，并使用 assert_frame_equal 方法进行断言
    unstacked = s.unstack(level=0)
    tm.assert_frame_equal(unstacked, expected.T)
    
    # 创建一个新的多级索引对象 index
    index = MultiIndex(
        levels=[["bar"], ["one", "two", "three"], [0, 1]],
        codes=[[0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],
    )
    
    # 创建一个 Series 对象 s，包含使用随机数生成器生成的标准正态分布数据，并使用 index 作为索引
    s = Series(np.random.default_rng(2).standard_normal(6), index=index)
    
    # 创建预期的 DataFrame 对象 expected，其中索引为 exp_index，包含 "bar" 列
    exp_index = MultiIndex(
        levels=[["one", "two", "three"], [0, 1]],
        codes=[[0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],
    )
    expected = DataFrame({"bar": s.values}, index=exp_index).sort_index(level=0)
    
    # 对 s 应用 unstack(0) 方法，得到展开的 DataFrame unstacked，并使用 assert_frame_equal 方法进行断言
    unstacked = s.unstack(0).sort_index()
    tm.assert_frame_equal(unstacked, expected)
    
    # GH5873 测试用例
    # 创建一个多级索引对象 idx
    idx = MultiIndex.from_arrays([[101, 102], [3.5, np.nan]])
    # 创建一个 Series 对象 ts，包含索引为 idx 的数据
    ts = Series([1, 2], index=idx)
    # 对 ts 应用 unstack 方法，得到展开的 DataFrame left
    left = ts.unstack()
    # 创建预期的 DataFrame 对象 right
    right = DataFrame(
        [[np.nan, 1], [2, np.nan]], index=[101, 102], columns=[np.nan, 3.5]
    )
    # 使用 assert_frame_equal 方法断言 left 和 right 相等
    tm.assert_frame_equal(left, right)
    
    # 创建一个多级索引对象 idx
    idx = MultiIndex.from_arrays(
        [
            ["cat", "cat", "cat", "dog", "dog"],
            ["a", "a", "b", "a", "b"],
            [1, 2, 1, 1, np.nan],
        ]
    )
    # 创建一个 Series 对象 ts，包含索引为 idx 的数据
    ts = Series([1.0, 1.1, 1.2, 1.3, 1.4], index=idx)
    # 创建预期的 DataFrame 对象 right，其中列为 ["cat", "dog"]
    right = DataFrame(
        [[1.0, 1.3], [1.1, np.nan], [np.nan, 1.4], [1.2, np.nan]],
        columns=["cat", "dog"],
    )
    # 创建新的多级索引对象 tpls，并将其作为 right 的索引
    tpls = [("a", 1), ("a", 2), ("b", np.nan), ("b", 1)]
    right.index = MultiIndex.from_tuples(tpls)
    # 对 ts 应用 unstack(level=0) 方法，得到展开的 DataFrame，并使用 assert_frame_equal 方法进行断言
    tm.assert_frame_equal(ts.unstack(level=0), right)


# 定义测试函数：测试 unstack 方法在多级索引中使用元组名称
def test_unstack_tuplename_in_multiindex():
    # 创建一个多级索引对象 idx，其中包含元组名称 ("A", "a")
    idx = MultiIndex.from_product(
        [["a", "b", "c"], [1, 2, 3]], names=[("A", "a"), ("B", "b")]
    )
    # 创建一个 Series 对象 ser，包含索引为 idx 的数据
    ser = Series(1, index=idx)
    # 对 ser 应用 unstack 方法，根据元组名称 ("A", "a") 展开数据
    result = ser.unstack(("A", "a"))
    
    # 创建预期的 DataFrame 对象 expected，其中列名为 MultiIndex，包含元组 ("A", "a")，索引为 ("B", "b")
    expected = DataFrame(
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        columns=MultiIndex.from_tuples([("a",), ("b",), ("c",)], names=[("A", "a")]),
        index=Index([1, 2, 3], name=("B", "b")),
    )
    # 使用 pandas._testing 模块的 assert_frame_equal 方法，断言 result 和 expected 相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例
@pytest.mark.parametrize(
    "unstack_idx, expected_values, expected_index, expected_columns",
    # 包含两个元组的列表，每个元组包括四个对象：
    #   - 第一个对象是一个元组 ("A", "a")
    #   - 第二个对象是一个二维列表，包含两个子列表，每个子列表包含两个整数 1
    #   - 第三个对象是一个 MultiIndex 对象，由元组列表构成，元组包括两个整数，分别为 (1, 3), (1, 4), (2, 3), (2, 4)，并有命名为 "B" 和 "C"
    #   - 第四个对象是一个 MultiIndex 对象，由元组列表构成，元组包括 ("a",), ("b",)，并有命名为 ("A", "a")
    (
        (
            ("A", "a"),
            [[1, 1], [1, 1], [1, 1], [1, 1]],
            MultiIndex.from_tuples([(1, 3), (1, 4), (2, 3), (2, 4)], names=["B", "C"]),
            MultiIndex.from_tuples([("a",), ("b",)], names=[("A", "a")]),
        ),
        # 包含两个元组的列表，每个元组包括四个对象：
        #   - 第一个对象是一个元组，由两个元素组成 ("A", "a") 和 "B"
        #   - 第二个对象是一个二维列表，包含两个子列表，每个子列表包含四个整数 1
        #   - 第三个对象是一个 Index 对象，包含两个整数 3 和 4，并有命名为 "C"
        #   - 第四个对象是一个 MultiIndex 对象，由元组列表构成，元组包括 ("a", 1), ("a", 2), ("b", 1), ("b", 2)，其中一个元组有两个元素，另一个有一个元素，命名为 ("A", "a") 和 "B"
        (
            (("A", "a"), "B"),
            [[1, 1, 1, 1], [1, 1, 1, 1]],
            Index([3, 4], name="C"),
            MultiIndex.from_tuples(
                [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=[("A", "a"), "B"]
            ),
        ),
    ],
# 定义测试函数，用于验证在多重索引中展开混合类型名称时的unstack方法
def test_unstack_mixed_type_name_in_multiindex(
    unstack_idx, expected_values, expected_index, expected_columns
):
    # GH 19966
    # 创建一个多重索引对象，其中包含三个级别，每个级别都有特定的名称
    idx = MultiIndex.from_product(
        [["a", "b"], [1, 2], [3, 4]], names=[("A", "a"), "B", "C"]
    )
    # 创建一个Series对象，所有条目的值为1，索引为上述创建的多重索引对象
    ser = Series(1, index=idx)
    # 对Series对象调用unstack方法，根据给定的unstack_idx参数展开数据
    result = ser.unstack(unstack_idx)

    # 创建一个期望的DataFrame对象，根据给定的预期值、索引和列
    expected = DataFrame(
        expected_values, columns=expected_columns, index=expected_index
    )
    # 使用pandas.testing模块中的assert_frame_equal函数比较result和expected，确保它们相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数，用于验证在多重索引中包含分类值时的unstack方法
def test_unstack_multi_index_categorical_values():
    # 创建一个DataFrame对象，其中元素值为正态分布的随机数
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    # 使用stack方法将DataFrame转换为Series，并重命名索引为["major", "minor"]
    mi = df.stack().index.rename(["major", "minor"])
    # 创建一个Series对象，所有条目的值为"foo"，索引为上述重命名后的多重索引对象，数据类型为分类
    ser = Series(["foo"] * len(mi), index=mi, name="category", dtype="category")

    # 对Series对象调用unstack方法，根据索引展开数据
    result = ser.unstack()

    # 获取Series对象的第一级别的唯一值构成的索引
    dti = ser.index.levels[0]
    # 创建一个分类对象，所有条目的值为"foo"
    c = pd.Categorical(["foo"] * len(dti))
    # 创建一个期望的DataFrame对象，根据给定的预期列名和索引名
    expected = DataFrame(
        {"A": c, "B": c, "C": c, "D": c},
        columns=Index(list("ABCD"), name="minor"),
        index=dti.rename("major"),
    )
    # 使用pandas.testing模块中的assert_frame_equal函数比较result和expected，确保它们相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数，用于验证在多重索引中混合级别名称时的unstack方法
def test_unstack_mixed_level_names():
    # GH#48763
    # 创建一个多重索引对象，其中包含三个级别，每个级别都有特定的名称
    arrays = [["a", "a"], [1, 2], ["red", "blue"]]
    idx = MultiIndex.from_arrays(arrays, names=("x", 0, "y"))
    # 创建一个Series对象，包含两个条目，索引为上述创建的多重索引对象
    ser = Series([1, 2], index=idx)
    # 对Series对象调用unstack方法，根据给定的"x"参数展开数据
    result = ser.unstack("x")
    # 创建一个期望的DataFrame对象，根据给定的预期值和多重索引
    expected = DataFrame(
        [[1], [2]],
        columns=Index(["a"], name="x"),
        index=MultiIndex.from_tuples([(1, "red"), (2, "blue")], names=[0, "y"]),
    )
    # 使用pandas.testing模块中的assert_frame_equal函数比较result和expected，确保它们相等
    tm.assert_frame_equal(result, expected)
```