# `D:\src\scipysrc\pandas\pandas\tests\frame\indexing\test_coercion.py`

```
"""
Tests for values coercion in setitem-like operations on DataFrame.

For the most part, these should be multi-column DataFrames, otherwise
we would share the tests with Series.
"""

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    NaT,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestDataFrameSetitemCoercion:
    @pytest.mark.parametrize("consolidate", [True, False])
    def test_loc_setitem_multiindex_columns(self, consolidate):
        # GH#18415 Setting values in a single column preserves dtype,
        #  while setting them in multiple columns did unwanted cast.

        # Note that A here has 2 blocks, below we do the same thing
        #  with a consolidated frame.
        
        # 创建一个形状为 (6, 5) 的全零 DataFrame，数据类型为 np.float32
        A = DataFrame(np.zeros((6, 5), dtype=np.float32))
        
        # 将 A 横向拼接一份，形成包含两个块的 DataFrame，分别用列名 1 和 2 标记
        A = pd.concat([A, A], axis=1, keys=[1, 2])
        
        # 如果 consolidate 为 True，则对 A 进行合并操作
        if consolidate:
            A = A._consolidate()

        # 在 A 中的 (1, slice(2, 3)) 区域的第 2 到第 3 行（含），设置为全一的 np.float32 数组
        A.loc[2:3, (1, slice(2, 3))] = np.ones((2, 2), dtype=np.float32)
        
        # 断言 A 中所有列的数据类型为 np.float32
        assert (A.dtypes == np.float32).all()

        # 在 A 中的 (1, slice(2, 3)) 区域的第 0 到第 5 行（含），设置为全一的 np.float32 数组
        A.loc[0:5, (1, slice(2, 3))] = np.ones((6, 2), dtype=np.float32)

        # 断言 A 中所有列的数据类型为 np.float32
        assert (A.dtypes == np.float32).all()

        # 在 A 的所有行中的 (1, slice(2, 3)) 区域，设置为全一的 np.float32 数组
        A.loc[:, (1, slice(2, 3))] = np.ones((6, 2), dtype=np.float32)
        
        # 断言 A 中所有列的数据类型为 np.float32
        assert (A.dtypes == np.float32).all()

        # TODO: i think this isn't about MultiIndex and could be done with iloc?


def test_37477():
    # fixed by GH#45121
    orig = DataFrame({"A": [1, 2, 3], "B": [3, 4, 5]})
    expected = DataFrame({"A": [1, 2, 3], "B": [3, 1.2, 5]})

    df = orig.copy()
    with tm.assert_produces_warning(
        FutureWarning, match="Setting an item of incompatible dtype"
    ):
        # 使用 .at 设置一个不兼容数据类型的项，期望触发 FutureWarning
        df.at[1, "B"] = 1.2
    tm.assert_frame_equal(df, expected)

    df = orig.copy()
    with tm.assert_produces_warning(
        FutureWarning, match="Setting an item of incompatible dtype"
    ):
        # 使用 .loc 设置一个不兼容数据类型的项，期望触发 FutureWarning
        df.loc[1, "B"] = 1.2
    tm.assert_frame_equal(df, expected)

    df = orig.copy()
    with tm.assert_produces_warning(
        FutureWarning, match="Setting an item of incompatible dtype"
    ):
        # 使用 .iat 设置一个不兼容数据类型的项，期望触发 FutureWarning
        df.iat[1, 1] = 1.2
    tm.assert_frame_equal(df, expected)

    df = orig.copy()
    with tm.assert_produces_warning(
        FutureWarning, match="Setting an item of incompatible dtype"
    ):
        # 使用 .iloc 设置一个不兼容数据类型的项，期望触发 FutureWarning
        df.iloc[1, 1] = 1.2
    tm.assert_frame_equal(df, expected)


def test_6942(indexer_al):
    # check that the .at __setitem__ after setting "Live" actually sets the data
    
    # 创建一个时间戳范围，从 start 开始的一个时间戳
    start = Timestamp("2014-04-01")
    t1 = Timestamp("2014-04-23 12:42:38.883082")
    t2 = Timestamp("2014-04-24 01:33:30.040039")

    # 创建一个以 start 为索引，列名为 ["timenow", "Live"] 的 DataFrame
    dti = date_range(start, periods=1)
    orig = DataFrame(index=dti, columns=["timenow", "Live"])

    df = orig.copy()
    
    # 使用 indexer_al 函数设置索引为 start，列名为 "timenow" 的值为 t1
    indexer_al(df)[start, "timenow"] = t1

    # 设置列 "Live" 的所有值为 True
    df["Live"] = True

    # 使用 .at 设置索引为 start，列名为 "timenow" 的值为 t2
    df.at[start, "timenow"] = t2
    
    # 断言 df 中第一行第一列的值等于 t2
    assert df.iloc[0, 0] == t2


def test_26395(indexer_al):
    # .at case fixed by GH#45121 (best guess)
    
    # 创建一个索引为 ["A", "B", "C"] 的空 DataFrame
    df = DataFrame(index=["A", "B", "C"])
    
    # 添加一列 "D"，并初始化为全零
    df["D"] = 0
    # 使用自定义的索引器 `indexer_al` 来设置 DataFrame `df` 中的元素 ("C", "D") 为整数 2
    indexer_al(df)["C", "D"] = 2
    
    # 创建一个预期的 DataFrame `expected`，包含列 "D" 的整数数据 [0, 0, 2]，并指定索引为 ["A", "B", "C"]，数据类型为 np.int64
    expected = DataFrame({"D": [0, 0, 2]}, index=["A", "B", "C"], dtype=np.int64)
    
    # 使用 `tm.assert_frame_equal` 来比较 DataFrame `df` 和预期的 DataFrame `expected` 是否相等
    tm.assert_frame_equal(df, expected)
    
    # 使用 `tm.assert_produces_warning` 确保下面的语句产生 FutureWarning 警告，并匹配 "Setting an item of incompatible dtype"
    with tm.assert_produces_warning(
        FutureWarning, match="Setting an item of incompatible dtype"
    ):
        # 使用索引器 `indexer_al` 将 DataFrame `df` 中的元素 ("C", "D") 设置为浮点数 44.5
        indexer_al(df)["C", "D"] = 44.5
    
    # 创建一个预期的 DataFrame `expected`，包含列 "D" 的数据 [0, 0, 44.5]，并指定索引为 ["A", "B", "C"]，数据类型为 np.float64
    expected = DataFrame({"D": [0, 0, 44.5]}, index=["A", "B", "C"], dtype=np.float64)
    
    # 再次使用 `tm.assert_frame_equal` 来比较 DataFrame `df` 和预期的 DataFrame `expected` 是否相等
    tm.assert_frame_equal(df, expected)
    
    # 使用 `tm.assert_produces_warning` 确保下面的语句产生 FutureWarning 警告，并匹配 "Setting an item of incompatible dtype"
    with tm.assert_produces_warning(
        FutureWarning, match="Setting an item of incompatible dtype"
    ):
        # 使用索引器 `indexer_al` 将 DataFrame `df` 中的元素 ("C", "D") 设置为字符串 "hello"
        indexer_al(df)["C", "D"] = "hello"
    
    # 创建一个预期的 DataFrame `expected`，包含列 "D" 的数据 [0, 0, "hello"]，并指定索引为 ["A", "B", "C"]，数据类型为 object
    expected = DataFrame({"D": [0, 0, "hello"]}, index=["A", "B", "C"], dtype=object)
    
    # 最后使用 `tm.assert_frame_equal` 来比较 DataFrame `df` 和预期的 DataFrame `expected` 是否相等
    tm.assert_frame_equal(df, expected)
@pytest.mark.xfail(reason="unwanted upcast")
# 标记此测试为预期失败，原因是不希望发生的类型提升

def test_15231():
    # 创建一个包含整数的 DataFrame，列名为 "a" 和 "b"
    df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    # 向 DataFrame 中添加一行，使用 Series 表示新行的数据
    df.loc[2] = Series({"a": 5, "b": 6})
    # 断言 DataFrame 的所有列的数据类型都是 np.int64
    assert (df.dtypes == np.int64).all()

    # 向 DataFrame 中添加一行，只指定部分列数据
    df.loc[3] = Series({"a": 7})

    # 断言 DataFrame 的 "a" 列不包含任何 NaN 值，不应该进行类型转换
    exp_dtypes = Series([np.int64, np.float64], dtype=object, index=["a", "b"])
    tm.assert_series_equal(df.dtypes, exp_dtypes)


def test_iloc_setitem_unnecesssary_float_upcasting():
    # GH#12255
    # 创建一个包含浮点数和字符串的 DataFrame
    df = DataFrame(
        {
            0: np.array([1, 3], dtype=np.float32),
            1: np.array([2, 4], dtype=np.float32),
            2: ["a", "b"],
        }
    )
    orig = df.copy()

    # 获取 DataFrame 第一列的值，并将其重新排列成二维数组
    values = df[0].values.reshape(2, 1)
    # 使用 iloc 将重新排列后的值赋值回 DataFrame 的第一列
    df.iloc[:, 0:1] = values

    # 断言修改后的 DataFrame 与原始 DataFrame 相等
    tm.assert_frame_equal(df, orig)


@pytest.mark.xfail(reason="unwanted casting to dt64")
# 标记此测试为预期失败，原因是不希望发生的类型转换为 dt64

def test_12499():
    # TODO: OP in GH#12499 used np.datetim64("NaT") instead of pd.NaT,
    #  which has consequences for the expected df["two"] (though i think at
    #  the time it might not have because of a separate bug). See if it makes
    #  a difference which one we use here.
    # 创建一个 Timestamp 对象
    ts = Timestamp("2016-03-01 03:13:22.98986", tz="UTC")

    # 创建一个包含一个字典的数据列表，并使用其创建 DataFrame
    data = [{"one": 0, "two": ts}]
    orig = DataFrame(data)
    df = orig.copy()
    # 向 DataFrame 中的第二行赋值，其中包含 NaN 值
    df.loc[1] = [np.nan, NaT]

    # 创建预期的 DataFrame，包含特定的 datetime64[ns, UTC] 类型的 Series
    expected = DataFrame(
        {"one": [0, np.nan], "two": Series([ts, NaT], dtype="datetime64[ns, UTC]")}
    )
    # 断言修改后的 DataFrame 与预期 DataFrame 相等
    tm.assert_frame_equal(df, expected)

    # 创建另一个数据列表，并使用其创建 DataFrame
    data = [{"one": 0, "two": ts}]
    df = orig.copy()
    # 向 DataFrame 的第二行全部列赋值，其中包含 NaN 值
    df.loc[1, :] = [np.nan, NaT]
    # 断言修改后的 DataFrame 与预期 DataFrame 相等
    tm.assert_frame_equal(df, expected)


def test_20476():
    # 创建一个 MultiIndex
    mi = MultiIndex.from_product([["A", "B"], ["a", "b", "c"]])
    # 创建一个填充了 -1 的 DataFrame，使用 MultiIndex 作为行和列索引
    df = DataFrame(-1, index=range(3), columns=mi)
    # 创建一个填充了整数和浮点数的 DataFrame
    filler = DataFrame([[1, 2, 3.0]] * 3, index=range(3), columns=["a", "b", "c"])
    # 将 filler DataFrame 赋值给 df 的 "A" 列
    df["A"] = filler

    # 创建预期的 DataFrame
    expected = DataFrame(
        {
            0: [1, 1, 1],
            1: [2, 2, 2],
            2: [3.0, 3.0, 3.0],
            3: [-1, -1, -1],
            4: [-1, -1, -1],
            5: [-1, -1, -1],
        }
    )
    expected.columns = mi
    # 创建预期的 Series，包含特定的数据类型
    exp_dtypes = Series(
        [np.dtype(np.int64)] * 2 + [np.dtype(np.float64)] + [np.dtype(np.int64)] * 3,
        index=mi,
    )
    # 断言 DataFrame 的数据类型与预期的数据类型 Series 相等
    tm.assert_series_equal(df.dtypes, exp_dtypes)
```