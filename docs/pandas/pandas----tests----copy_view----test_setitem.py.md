# `D:\src\scipysrc\pandas\pandas\tests\copy_view\test_setitem.py`

```
# 导入 NumPy 库，用于处理数组和数值计算
import numpy as np

# 从 pandas 库中导入多个类和函数，包括 DataFrame、Index、MultiIndex、RangeIndex 和 Series
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
)

# 从 pandas._testing 模块导入 tm 对象，用于测试和断言
import pandas._testing as tm

# 从 pandas.tests.copy_view.util 模块中导入 get_array 函数，用于获取对象的数组表示
from pandas.tests.copy_view.util import get_array

# -----------------------------------------------------------------------------
# DataFrame 中设置值时的复制/视图行为


def test_set_column_with_array():
    # Case: 将数组作为新列设置 (df[col] = arr)，此操作会复制数据
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    arr = np.array([1, 2, 3], dtype="int64")

    df["c"] = arr

    # 数组数据已复制
    assert not np.shares_memory(get_array(df, "c"), arr)
    # 因此修改数组不会修改 DataFrame
    arr[0] = 0
    tm.assert_series_equal(df["c"], Series([1, 2, 3], name="c"))


def test_set_column_with_series():
    # Case: 将 Series 作为新列设置 (df[col] = s)，此操作会复制数据（使用 CoW 延迟复制）
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    ser = Series([1, 2, 3])

    df["c"] = ser

    assert np.shares_memory(get_array(df, "c"), get_array(ser))

    # 修改 Series 不会修改 DataFrame
    ser.iloc[0] = 0
    assert ser.iloc[0] == 0
    tm.assert_series_equal(df["c"], Series([1, 2, 3], name="c"))


def test_set_column_with_index():
    # Case: 将 Index 作为新列设置 (df[col] = idx)，此操作会复制数据
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    idx = Index([1, 2, 3])

    df["c"] = idx

    # 索引数据已复制
    assert not np.shares_memory(get_array(df, "c"), idx.values)

    idx = RangeIndex(1, 4)
    arr = idx.values

    df["d"] = idx

    assert not np.shares_memory(get_array(df, "d"), arr)


def test_set_columns_with_dataframe():
    # Case: 将 DataFrame 作为新列设置，此操作会复制数据（使用 CoW 延迟复制）
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]})

    df[["c", "d"]] = df2

    assert np.shares_memory(get_array(df, "c"), get_array(df2, "c"))
    # 修改设置的 DataFrame 不会修改原始 DataFrame
    df2.iloc[0, 0] = 0
    tm.assert_series_equal(df["c"], Series([7, 8, 9], name="c"))


def test_setitem_series_no_copy():
    # Case: 将 Series 作为列设置到 DataFrame 中，可能会延迟复制数据
    df = DataFrame({"a": [1, 2, 3]})
    rhs = Series([4, 5, 6])
    rhs_orig = rhs.copy()

    # 添加一个新列
    df["b"] = rhs
    assert np.shares_memory(get_array(rhs), get_array(df, "b"))

    df.iloc[0, 1] = 100
    tm.assert_series_equal(rhs, rhs_orig)


def test_setitem_series_no_copy_single_block():
    # 覆盖一个单块的现有列
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    rhs = Series([4, 5, 6])
    rhs_orig = rhs.copy()

    df["a"] = rhs
    assert np.shares_memory(get_array(rhs), get_array(df, "a"))

    df.iloc[0, 0] = 100
    tm.assert_series_equal(rhs, rhs_orig)
# 在一个较大的块中覆盖已存在的列
def test_setitem_series_no_copy_split_block():
    # 创建一个 DataFrame，包含两列 "a" 和 "b"，"b" 初始值为标量 1
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    # 创建一个 Series 对象 rhs，包含值 [4, 5, 6]
    rhs = Series([4, 5, 6])
    # 备份 rhs，用于后续的比较
    rhs_orig = rhs.copy()

    # 将 Series rhs 赋值给 DataFrame df 的列 "b"
    df["b"] = rhs
    # 断言 rhs 和 df["b"] 共享内存
    assert np.shares_memory(get_array(rhs), get_array(df, "b"))

    # 修改 DataFrame df 的第一行第二列的值为 100
    df.iloc[0, 1] = 100
    # 断言 rhs 未被修改，与备份 rhs_orig 相等
    tm.assert_series_equal(rhs, rhs_orig)


# 设置一个 Series 到多列会重复数据
# （目前是急切地复制数据）
def test_setitem_series_column_midx_broadcasting():
    # 创建一个 DataFrame，包含多层索引，列值为 [[1, 2, 3], [3, 4, 5]]，列标签为 ["a", "a", "b"] 和 [1, 2, 3]
    df = DataFrame(
        [[1, 2, 3], [3, 4, 5]],
        columns=MultiIndex.from_arrays([["a", "a", "b"], [1, 2, 3]]),
    )
    # 创建一个 Series 对象 rhs，包含值 [10, 11]
    rhs = Series([10, 11])
    # 将 Series rhs 赋值给 DataFrame df 的列 "a"
    df["a"] = rhs
    # 断言 rhs 和 df 的第一列共享内存
    assert not np.shares_memory(get_array(rhs), df._get_column_array(0))
    # 断言 df 的内部管理对象没有引用第一列
    assert df._mgr._has_no_reference(0)


# 使用原位操作符设置列
def test_set_column_with_inplace_operator():
    # 创建一个 DataFrame，包含两列 "a" 和 "b"，分别为 [1, 2, 3] 和 [4, 5, 6]
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # 使用 with 语句断言操作不会产生任何警告
    with tm.assert_produces_warning(None):
        # 将 DataFrame df 的列 "a" 全部加 1
        df["a"] += 1

    # 当不在链式操作中时，应产生警告
    # 重新创建 DataFrame df，包含两列 "a" 和 "b"，分别为 [1, 2, 3] 和 [4, 5, 6]
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # 创建一个 Series 对象 ser，引用 DataFrame df 的列 "a"
    ser = df["a"]
    # 对 Series ser 全部加 1
    ser += 1
```