# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_sort_values.py`

```
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    NaT,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.util.version import Version

class TestDataFrameSortValues:
    @pytest.mark.parametrize("dtype", [np.uint8, bool])
    def test_sort_values_sparse_no_warning(self, dtype):
        # GH#45618: 测试用例标识，用于跟踪问题编号
        # 创建一个包含分类数据的序列
        ser = pd.Series(Categorical(["a", "b", "a"], categories=["a", "b", "c"]))
        # 使用 pd.get_dummies 函数生成稀疏DataFrame
        df = pd.get_dummies(ser, dtype=dtype, sparse=True)

        with tm.assert_produces_warning(None):
            # 使用 df.columns.tolist() 生成的列名列表对 DataFrame 进行排序
            # 在这个测试中，我们希望不会收到关于从 SparseArray 构建索引的警告
            df.sort_values(by=df.columns.tolist())
    # 定义测试函数，用于测试排序功能
    def test_sort_values(self):
        # 创建一个 DataFrame 对象
        frame = DataFrame(
            [[1, 1, 2], [3, 1, 0], [4, 5, 6]], index=[1, 2, 3], columns=list("ABC")
        )

        # 按列排序（axis=0）
        sorted_df = frame.sort_values(by="A")
        # 获取按列排序后的索引顺序
        indexer = frame["A"].argsort().values
        # 按照索引顺序重新排序原始 DataFrame
        expected = frame.loc[frame.index[indexer]]
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(sorted_df, expected)

        # 按列排序，降序排列
        sorted_df = frame.sort_values(by="A", ascending=False)
        # 反转索引顺序
        indexer = indexer[::-1]
        # 按照反转后的索引顺序重新排序原始 DataFrame
        expected = frame.loc[frame.index[indexer]]
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(sorted_df, expected)

        # 再次按列排序，降序排列
        sorted_df = frame.sort_values(by="A", ascending=False)
        # 断言两个 DataFrame 是否相等（此处与前一次排序重复，可能是错误的复制粘贴）
        tm.assert_frame_equal(sorted_df, expected)

        # GH4839
        # 按列排序，降序排列，使用列表形式指定列名和排序顺序
        sorted_df = frame.sort_values(by=["A"], ascending=[False])
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(sorted_df, expected)

        # 多列排序
        sorted_df = frame.sort_values(by=["B", "C"])
        # 按指定顺序重新索引原始 DataFrame
        expected = frame.loc[[2, 1, 3]]
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(sorted_df, expected)

        # 多列排序，降序排列
        sorted_df = frame.sort_values(by=["B", "C"], ascending=False)
        # 按照反转后的索引顺序重新排序原始 DataFrame
        tm.assert_frame_equal(sorted_df, expected[::-1])

        # 多列排序，分别指定升序和降序
        sorted_df = frame.sort_values(by=["B", "A"], ascending=[True, False])
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(sorted_df, expected)

        # 测试抛出异常，尝试在不存在的轴向上排序
        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            frame.sort_values(by=["A", "B"], axis=2, inplace=True)

        # 按行排序（axis=1），GH#10806
        sorted_df = frame.sort_values(by=3, axis=1)
        # 期望结果是原始的 frame
        expected = frame
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(sorted_df, expected)

        # 按行排序，降序排列
        sorted_df = frame.sort_values(by=3, axis=1, ascending=False)
        # 按指定顺序重新索引原始 DataFrame 的列
        expected = frame.reindex(columns=["C", "B", "A"])
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(sorted_df, expected)

        # 按多个列名排序（axis="columns"）
        sorted_df = frame.sort_values(by=[1, 2], axis="columns")
        # 按指定顺序重新索引原始 DataFrame 的列
        expected = frame.reindex(columns=["B", "A", "C"])
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(sorted_df, expected)

        # 按多个列名排序，分别指定升序和降序
        sorted_df = frame.sort_values(by=[1, 3], axis=1, ascending=[True, False])
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(sorted_df, expected)

        # 按多个列名排序，全部降序排列
        sorted_df = frame.sort_values(by=[1, 3], axis=1, ascending=False)
        # 按指定顺序重新索引原始 DataFrame 的列
        expected = frame.reindex(columns=["C", "B", "A"])
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(sorted_df, expected)

        # 测试抛出异常，长度不匹配的升序参数
        msg = r"Length of ascending \(5\) != length of by \(2\)"
        with pytest.raises(ValueError, match=msg):
            frame.sort_values(by=["A", "B"], axis=0, ascending=[True] * 5)

    # 测试空列表排序
    def test_sort_values_by_empty_list(self):
        # 创建一个期望的 DataFrame
        expected = DataFrame({"a": [1, 4, 2, 5, 3, 6]})
        # 对空列表进行排序
        result = expected.sort_values(by=[])
        # 断言排序结果与期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)
        # 检查结果和期望结果不是同一个对象
        assert result is not expected
    def test_sort_values_inplace(self):
        # 创建一个 DataFrame 对象，其中包含从标准正态分布中随机生成的数据，4行4列
        frame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[1, 2, 3, 4],
            columns=["A", "B", "C", "D"],
        )

        # 复制 DataFrame 对象，以便进行排序操作，inplace=True 表示在原对象上进行排序
        sorted_df = frame.copy()
        # 对 DataFrame 按列 'A' 进行排序，返回值为 None（因为 inplace=True）
        return_value = sorted_df.sort_values(by="A", inplace=True)
        # 断言排序返回值为 None
        assert return_value is None
        # 生成预期的排序后的 DataFrame
        expected = frame.sort_values(by="A")
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(sorted_df, expected)

        # 复制 DataFrame 对象
        sorted_df = frame.copy()
        # 按照列索引 1（即 'B' 列）对 DataFrame 进行排序，axis=1 表示按列排序
        return_value = sorted_df.sort_values(by=1, axis=1, inplace=True)
        # 断言排序返回值为 None
        assert return_value is None
        # 生成预期的排序后的 DataFrame
        expected = frame.sort_values(by=1, axis=1)
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(sorted_df, expected)

        # 复制 DataFrame 对象
        sorted_df = frame.copy()
        # 按列 'A' 降序排序
        return_value = sorted_df.sort_values(by="A", ascending=False, inplace=True)
        # 断言排序返回值为 None
        assert return_value is None
        # 生成预期的排序后的 DataFrame
        expected = frame.sort_values(by="A", ascending=False)
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(sorted_df, expected)

        # 复制 DataFrame 对象
        sorted_df = frame.copy()
        # 按多列 ['A', 'B'] 的值降序排序
        return_value = sorted_df.sort_values(
            by=["A", "B"], ascending=False, inplace=True
        )
        # 断言排序返回值为 None
        assert return_value is None
        # 生成预期的排序后的 DataFrame
        expected = frame.sort_values(by=["A", "B"], ascending=False)
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_multicolumn(self):
        # 创建包含多个重复列 'A' 和 'B' 的数组，以及随机列 'C' 的 DataFrame
        A = np.arange(5).repeat(20)
        B = np.tile(np.arange(5), 20)
        np.random.default_rng(2).shuffle(A)
        np.random.default_rng(2).shuffle(B)
        frame = DataFrame(
            {"A": A, "B": B, "C": np.random.default_rng(2).standard_normal(100)}
        )

        # 按照多列 ['A', 'B'] 的值排序 DataFrame
        result = frame.sort_values(by=["A", "B"])
        # 使用 np.lexsort 进行多列排序的索引排序
        indexer = np.lexsort((frame["B"], frame["A"]))
        # 根据索引获取预期的排序后 DataFrame
        expected = frame.take(indexer)
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 按照多列 ['A', 'B'] 的值降序排序 DataFrame
        result = frame.sort_values(by=["A", "B"], ascending=False)
        # 使用 np.lexsort 和 rank 方法进行多列降序排序的索引排序
        indexer = np.lexsort(
            (frame["B"].rank(ascending=False), frame["A"].rank(ascending=False))
        )
        # 根据索引获取预期的排序后 DataFrame
        expected = frame.take(indexer)
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 按照多列 ['B', 'A'] 的值排序 DataFrame
        result = frame.sort_values(by=["B", "A"])
        # 使用 np.lexsort 进行多列排序的索引排序
        indexer = np.lexsort((frame["A"], frame["B"]))
        # 根据索引获取预期的排序后 DataFrame
        expected = frame.take(indexer)
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_sort_values_multicolumn_uint64(self):
        # GH#9918
        # 创建包含 uint64 类型列 'a' 和 'b' 的 DataFrame

        df = DataFrame(
            {
                "a": pd.Series([18446637057563306014, 1162265347240853609]),
                "b": pd.Series([1, 2]),
            }
        )
        # 将 'a' 列的数据类型转换为 uint64
        df["a"] = df["a"].astype(np.uint64)
        # 按照多列 ['a', 'b'] 的值排序 DataFrame
        result = df.sort_values(["a", "b"])

        # 创建预期的排序后的 DataFrame
        expected = DataFrame(
            {
                "a": pd.Series([18446637057563306014, 1162265347240853609]),
                "b": pd.Series([1, 2]),
            },
            index=pd.Index([1, 0]),
        )

        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
    def test_sort_values_nan(self):
        # GH#3917
        # 创建包含数值和NaN的DataFrame
        df = DataFrame(
            {"A": [1, 2, np.nan, 1, 6, 8, 4], "B": [9, np.nan, 5, 2, 5, 4, 5]}
        )

        # sort one column only
        # 期望的排序结果DataFrame，按照"A"列排序，NaN值放在最前面
        expected = DataFrame(
            {"A": [np.nan, 1, 1, 2, 4, 6, 8], "B": [5, 9, 2, np.nan, 5, 5, 4]},
            index=[2, 0, 3, 1, 6, 4, 5],
        )
        sorted_df = df.sort_values(["A"], na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # 期望的排序结果DataFrame，按照"A"列排序，NaN值放在最前面，降序排列
        expected = DataFrame(
            {"A": [np.nan, 8, 6, 4, 2, 1, 1], "B": [5, 4, 5, 5, np.nan, 9, 2]},
            index=[2, 5, 4, 6, 1, 0, 3],
        )
        sorted_df = df.sort_values(["A"], na_position="first", ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

        # 重新按照指定列顺序排列的DataFrame
        expected = df.reindex(columns=["B", "A"])
        sorted_df = df.sort_values(by=1, axis=1, na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='last'，按照默认顺序排序的期望DataFrame
        expected = DataFrame(
            {"A": [1, 1, 2, 4, 6, 8, np.nan], "B": [2, 9, np.nan, 5, 5, 4, 5]},
            index=[3, 0, 1, 6, 4, 5, 2],
        )
        sorted_df = df.sort_values(["A", "B"])
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='first'，按照指定顺序排序的期望DataFrame
        expected = DataFrame(
            {"A": [np.nan, 1, 1, 2, 4, 6, 8], "B": [5, 2, 9, np.nan, 5, 5, 4]},
            index=[2, 3, 0, 1, 6, 4, 5],
        )
        sorted_df = df.sort_values(["A", "B"], na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='first'，按照指定顺序和降序排列的期望DataFrame
        expected = DataFrame(
            {"A": [np.nan, 1, 1, 2, 4, 6, 8], "B": [5, 9, 2, np.nan, 5, 5, 4]},
            index=[2, 0, 3, 1, 6, 4, 5],
        )
        sorted_df = df.sort_values(["A", "B"], ascending=[1, 0], na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='last'，按照指定顺序和降序排列的期望DataFrame
        expected = DataFrame(
            {"A": [8, 6, 4, 2, 1, 1, np.nan], "B": [4, 5, 5, np.nan, 2, 9, 5]},
            index=[5, 4, 6, 1, 3, 0, 2],
        )
        sorted_df = df.sort_values(["A", "B"], ascending=[0, 1], na_position="last")
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_stable_descending_sort(self):
        # GH#6399
        # 创建包含列"sort_col"和"order"的DataFrame
        df = DataFrame(
            [[2, "first"], [2, "second"], [1, "a"], [1, "b"]],
            columns=["sort_col", "order"],
        )
        # 按照"sort_col"列的值，使用稳定的mergesort算法进行降序排序
        sorted_df = df.sort_values(by="sort_col", kind="mergesort", ascending=False)
        # 断言原始DataFrame和排序后DataFrame相等
        tm.assert_frame_equal(df, sorted_df)
    @pytest.mark.parametrize(
        "expected_idx_non_na, ascending",
        [
            # 定义参数化测试的参数，包括预期非 NaN 索引和排序顺序
            [
                [3, 4, 5, 0, 1, 8, 6, 9, 7, 10, 13, 14],
                [True, True],
            ],
            [
                [0, 3, 4, 5, 1, 8, 6, 7, 10, 13, 14, 9],
                [True, False],
            ],
            [
                [9, 7, 10, 13, 14, 6, 8, 1, 3, 4, 5, 0],
                [False, True],
            ],
            [
                [7, 10, 13, 14, 9, 6, 8, 1, 0, 3, 4, 5],
                [False, False],
            ],
        ],
    )
    @pytest.mark.parametrize("na_position", ["first", "last"])
    # 定义参数化测试函数，测试多列排序时的稳定性
    def test_sort_values_stable_multicolumn_sort(
        self, expected_idx_non_na, ascending, na_position
    ):
        # GH#38426 清楚说明多列/标签排序时 sort_values 的稳定性
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 1, 1, 1, 6, 8, 4, 8, 8, np.nan, np.nan, 8, 8],
                "B": [9, np.nan, 5, 2, 2, 2, 5, 4, 5, 3, 4, np.nan, np.nan, 4, 4],
            }
        )
        # 对于列 "B" 中所有含有 NaN 的行，列 "A" 中只有唯一值，因此，只有含有 NaN 的行需要单独处理：
        expected_idx = (
            [11, 12, 2] + expected_idx_non_na
            if na_position == "first"
            else expected_idx_non_na + [2, 11, 12]
        )
        expected = df.take(expected_idx)
        # 对 DataFrame 进行排序，并验证排序后的结果是否与预期一致
        sorted_df = df.sort_values(
            ["A", "B"], ascending=ascending, na_position=na_position
        )
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_stable_categorial(self):
        # GH#16793
        # 创建一个包含分类数据的 DataFrame
        df = DataFrame({"x": Categorical(np.repeat([1, 2, 3, 4], 5), ordered=True)})
        expected = df.copy()
        # 使用 mergesort 算法对列 "x" 进行排序
        sorted_df = df.sort_values("x", kind="mergesort")
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_datetimes(self):
        # GH#3461, argsort / lexsort differences for a datetime column
        # 创建一个包含日期时间索引的 DataFrame
        df = DataFrame(
            ["a", "a", "a", "b", "c", "d", "e", "f", "g"],
            columns=["A"],
            index=date_range("20130101", periods=9),
        )
        # 创建日期时间对象列表
        dts = [
            Timestamp(x)
            for x in [
                "2004-02-11",
                "2004-01-21",
                "2004-01-26",
                "2005-09-20",
                "2010-10-04",
                "2009-05-12",
                "2008-11-12",
                "2010-09-28",
                "2010-09-28",
            ]
        ]
        df["B"] = dts[::2] + dts[1::2]
        df["C"] = 2.0
        df["A1"] = 3.0

        # 对列 "A" 进行排序，并验证按单列和多列排序的结果是否相同
        df1 = df.sort_values(by="A")
        df2 = df.sort_values(by=["A"])
        tm.assert_frame_equal(df1, df2)

        # 对列 "B" 进行排序，并验证按单列和多列排序的结果是否相同
        df1 = df.sort_values(by="B")
        df2 = df.sort_values(by=["B"])
        tm.assert_frame_equal(df1, df2)

        df1 = df.sort_values(by="B")

        # 对列 "B" 和 "C" 进行排序，并验证结果是否与按列 "B" 单独排序的结果相同
        df2 = df.sort_values(by=["C", "B"])
        tm.assert_frame_equal(df1, df2)
    # 定义一个测试方法，测试在 DataFrame 列上进行原地排序时的异常情况处理
    def test_sort_values_frame_column_inplace_sort_exception(self, float_frame):
        # 获取 DataFrame 列 "A" 的 Series 对象 s
        s = float_frame["A"]
        # 复制整个 float_frame 数据框，备份原始数据
        float_frame_orig = float_frame.copy()
        # 提示信息：Series 是一个新对象，因此可以原地修改而不影响原始数据框
        # INFO(CoW) Series is a new object, so can be changed inplace
        # without modifying original datafame
        # 在原地进行 Series s 的排序操作
        s.sort_values(inplace=True)
        # 断言：验证排序后的 Series s 与排序后的 float_frame_orig["A"] 相等
        tm.assert_series_equal(s, float_frame_orig["A"].sort_values())
        # 断言：验证整个数据框 float_frame 没有发生变化
        tm.assert_frame_equal(float_frame, float_frame_orig)

        # 复制 Series s 到 cp
        cp = s.copy()
        # 在不使用 inplace 参数的情况下进行排序，仅验证是否正常工作
        cp.sort_values()  # it works!

    # 定义一个测试方法，测试在整数列中处理 NaT 值时的排序行为
    def test_sort_values_nat_values_in_int_column(self):
        # GH#14922: "sorting with large float and multiple columns incorrect"

        # 问题原因是 int64 类型的 NaT 被视为 "na"，这只对 datetime64 列才是正确的行为

        # 定义整数和浮点数列的初始值
        int_values = (2, int(NaT._value))
        float_values = (2.0, -1.797693e308)

        # 创建一个 DataFrame 包含 int 和 float 列
        df = DataFrame(
            {"int": int_values, "float": float_values}, columns=["int", "float"]
        )

        # 创建一个按逆序排列的 DataFrame
        df_reversed = DataFrame(
            {"int": int_values[::-1], "float": float_values[::-1]},
            columns=["int", "float"],
            index=[1, 0],
        )

        # NaT 对于 int64 列不是 "na"，因此 na_position 参数不应影响结果：
        # 按指定的列排序，na_position 设置为 "last"
        df_sorted = df.sort_values(["int", "float"], na_position="last")
        # 断言：验证排序后的 DataFrame df_sorted 与预期的 df_reversed 相等
        tm.assert_frame_equal(df_sorted, df_reversed)

        # 按指定的列排序，na_position 设置为 "first"
        df_sorted = df.sort_values(["int", "float"], na_position="first")
        # 断言：验证排序后的 DataFrame df_sorted 与预期的 df_reversed 相等
        tm.assert_frame_equal(df_sorted, df_reversed)

        # 反向排序顺序
        df_sorted = df.sort_values(["int", "float"], ascending=False)
        # 断言：验证排序后的 DataFrame df_sorted 与原始 df 相等
        tm.assert_frame_equal(df_sorted, df)

        # 现在检查 NaT 对于 datetime64 列是否仍然被视为 "na"
        df = DataFrame(
            {"datetime": [Timestamp("2016-01-01"), NaT], "float": float_values},
            columns=["datetime", "float"],
        )

        df_reversed = DataFrame(
            {"datetime": [NaT, Timestamp("2016-01-01")], "float": float_values[::-1]},
            columns=["datetime", "float"],
            index=[1, 0],
        )

        # 按指定的列排序，na_position 设置为 "first"
        df_sorted = df.sort_values(["datetime", "float"], na_position="first")
        # 断言：验证排序后的 DataFrame df_sorted 与预期的 df_reversed 相等
        tm.assert_frame_equal(df_sorted, df_reversed)

        # 按指定的列排序，na_position 设置为 "last"
        df_sorted = df.sort_values(["datetime", "float"], na_position="last")
        # 断言：验证排序后的 DataFrame df_sorted 与原始 df 相等
        tm.assert_frame_equal(df_sorted, df)

        # 升序排序不应影响结果
        df_sorted = df.sort_values(["datetime", "float"], ascending=False)
        # 断言：验证排序后的 DataFrame df_sorted 与原始 df 相等
        tm.assert_frame_equal(df_sorted, df)
    def test_sort_nat(self):
        # GH 16836
        # 定义列表 d1，包含 Timestamp 对象，代表日期 ["2016-01-01", "2015-01-01", np.nan, "2016-01-01"]
        d1 = [Timestamp(x) for x in ["2016-01-01", "2015-01-01", np.nan, "2016-01-01"]]
        # 定义列表 d2，包含 Timestamp 对象，代表日期 ["2017-01-01", "2014-01-01", "2016-01-01", "2015-01-01"]
        d2 = [
            Timestamp(x)
            for x in ["2017-01-01", "2014-01-01", "2016-01-01", "2015-01-01"]
        ]
        # 创建 DataFrame df，包含列 'a' 和 'b'，索引为 [0, 1, 2, 3]
        df = DataFrame({"a": d1, "b": d2}, index=[0, 1, 2, 3])

        # 定义列表 d3，包含 Timestamp 对象，代表日期 ["2015-01-01", "2016-01-01", "2016-01-01", np.nan]
        d3 = [Timestamp(x) for x in ["2015-01-01", "2016-01-01", "2016-01-01", np.nan]]
        # 定义列表 d4，包含 Timestamp 对象，代表日期 ["2014-01-01", "2015-01-01", "2017-01-01", "2016-01-01"]
        d4 = [
            Timestamp(x)
            for x in ["2014-01-01", "2015-01-01", "2017-01-01", "2016-01-01"]
        ]
        # 创建期望的 DataFrame expected，包含列 'a' 和 'b'，索引为 [1, 3, 0, 2]
        expected = DataFrame({"a": d3, "b": d4}, index=[1, 3, 0, 2])
        
        # 对 DataFrame df 根据列 'a' 和 'b' 进行自然排序，生成排序后的 DataFrame sorted_df
        sorted_df = df.sort_values(by=["a", "b"])
        # 使用测试框架进行排序后的 DataFrame sorted_df 和期望的 DataFrame expected 的比较
        tm.assert_frame_equal(sorted_df, expected)
    def test_sort_values_na_position_with_categories(self):
        # 测试函数：test_sort_values_na_position_with_categories
        # GH#22556
        # Issue编号 GH#22556，用于跟踪问题
        # Positioning missing value properly when column is Categorical.
        # 当列为分类变量时，正确定位缺失值的位置

        # 定义分类变量和缺失值的索引
        categories = ["A", "B", "C"]
        category_indices = [0, 2, 4]
        list_of_nans = [np.nan, np.nan]
        na_indices = [1, 3]
        na_position_first = "first"
        na_position_last = "last"
        column_name = "c"

        # 对分类变量和索引进行逆序排序
        reversed_categories = sorted(categories, reverse=True)
        reversed_category_indices = sorted(category_indices, reverse=True)
        reversed_na_indices = sorted(na_indices)

        # 创建包含分类变量和缺失值的数据框
        df = DataFrame(
            {
                column_name: Categorical(
                    ["A", np.nan, "B", np.nan, "C"], categories=categories, ordered=True
                )
            }
        )

        # 按照指定的条件排序数据框，na放在最前面
        result = df.sort_values(
            by=column_name, ascending=True, na_position=na_position_first
        )
        # 期望的排序结果
        expected = DataFrame(
            {
                column_name: Categorical(
                    list_of_nans + categories, categories=categories, ordered=True
                )
            },
            index=na_indices + category_indices,
        )
        # 断言排序后的结果与期望结果相等
        tm.assert_frame_equal(result, expected)

        # 按照指定的条件排序数据框，na放在最后面
        result = df.sort_values(
            by=column_name, ascending=True, na_position=na_position_last
        )
        # 期望的排序结果
        expected = DataFrame(
            {
                column_name: Categorical(
                    categories + list_of_nans, categories=categories, ordered=True
                )
            },
            index=category_indices + na_indices,
        )
        # 断言排序后的结果与期望结果相等
        tm.assert_frame_equal(result, expected)

        # 按照指定的条件倒序排序数据框，na放在最前面
        result = df.sort_values(
            by=column_name, ascending=False, na_position=na_position_first
        )
        # 期望的排序结果
        expected = DataFrame(
            {
                column_name: Categorical(
                    list_of_nans + reversed_categories,
                    categories=categories,
                    ordered=True,
                )
            },
            index=reversed_na_indices + reversed_category_indices,
        )
        # 断言排序后的结果与期望结果相等
        tm.assert_frame_equal(result, expected)

        # 按照指定的条件倒序排序数据框，na放在最后面
        result = df.sort_values(
            by=column_name, ascending=False, na_position=na_position_last
        )
        # 期望的排序结果
        expected = DataFrame(
            {
                column_name: Categorical(
                    reversed_categories + list_of_nans,
                    categories=categories,
                    ordered=True,
                )
            },
            index=reversed_category_indices + reversed_na_indices,
        )
        # 断言排序后的结果与期望结果相等
        tm.assert_frame_equal(result, expected)
    def test_sort_values_nat(self):
        # GH#16836
        # 创建日期列表 d1 和 d2，并使用 Timestamp 将日期字符串转换为时间戳对象
        d1 = [Timestamp(x) for x in ["2016-01-01", "2015-01-01", np.nan, "2016-01-01"]]
        d2 = [
            Timestamp(x)
            for x in ["2017-01-01", "2014-01-01", "2016-01-01", "2015-01-01"]
        ]
        # 创建 DataFrame 对象 df，包含两列 "a" 和 "b"，使用 d1 和 d2 作为数据，指定索引为 [0, 1, 2, 3]
        df = DataFrame({"a": d1, "b": d2}, index=[0, 1, 2, 3])

        # 创建日期列表 d3 和 d4，使用 Timestamp 将日期字符串转换为时间戳对象
        d3 = [Timestamp(x) for x in ["2015-01-01", "2016-01-01", "2016-01-01", np.nan]]
        d4 = [
            Timestamp(x)
            for x in ["2014-01-01", "2015-01-01", "2017-01-01", "2016-01-01"]
        ]
        # 创建期望的 DataFrame 对象 expected，包含两列 "a" 和 "b"，使用 d3 和 d4 作为数据，指定索引为 [1, 3, 0, 2]
        expected = DataFrame({"a": d3, "b": d4}, index=[1, 3, 0, 2])

        # 对 DataFrame df 按列 "a" 和 "b" 进行排序，生成排序后的 DataFrame sorted_df
        sorted_df = df.sort_values(by=["a", "b"])
        
        # 使用 assert_frame_equal 检查 sorted_df 是否与期望的 expected DataFrame 相等
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_na_position_with_categories_raises(self):
        # 创建包含分类数据的 DataFrame df，其中列 "c" 使用 Categorical 对象
        df = DataFrame(
            {
                "c": Categorical(
                    ["A", np.nan, "B", np.nan, "C"],
                    categories=["A", "B", "C"],
                    ordered=True,
                )
            }
        )

        # 使用 pytest 的 raises 方法检查在排序时使用了无效的 na_position 参数
        with pytest.raises(ValueError, match="invalid na_position: bad_position"):
            df.sort_values(by="c", ascending=False, na_position="bad_position")

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize(
        "original_dict, sorted_dict, ignore_index, output_index",
        [
            ({"A": [1, 2, 3]}, {"A": [3, 2, 1]}, True, [0, 1, 2]),
            ({"A": [1, 2, 3]}, {"A": [3, 2, 1]}, False, [2, 1, 0]),
            (
                {"A": [1, 2, 3], "B": [2, 3, 4]},
                {"A": [3, 2, 1], "B": [4, 3, 2]},
                True,
                [0, 1, 2],
            ),
            (
                {"A": [1, 2, 3], "B": [2, 3, 4]},
                {"A": [3, 2, 1], "B": [4, 3, 2]},
                False,
                [2, 1, 0],
            ),
        ],
    )
    def test_sort_values_ignore_index(
        self, inplace, original_dict, sorted_dict, ignore_index, output_index
    ):
        # GH 30114
        # 创建包含 original_dict 数据的 DataFrame df
        df = DataFrame(original_dict)
        # 创建期望的 DataFrame expected，包含 sorted_dict 数据和指定的 output_index 索引
        expected = DataFrame(sorted_dict, index=output_index)
        kwargs = {"ignore_index": ignore_index, "inplace": inplace}

        if inplace:
            # 如果 inplace 为 True，复制 df 并在副本上应用 sort_values 方法
            result_df = df.copy()
            result_df.sort_values("A", ascending=False, **kwargs)
        else:
            # 如果 inplace 为 False，在 df 上直接应用 sort_values 方法
            result_df = df.sort_values("A", ascending=False, **kwargs)

        # 使用 assert_frame_equal 检查 result_df 是否与期望的 expected DataFrame 相等
        tm.assert_frame_equal(result_df, expected)
        # 使用 assert_frame_equal 检查排序操作是否未修改原始的 df
        tm.assert_frame_equal(df, DataFrame(original_dict))
    # 定义一个测试方法，用于测试自然排序和默认的 NA 位置行为
    def test_sort_values_nat_na_position_default(self):
        # GH 13230: Issue reference for tracking purposes
        # 创建一个预期的 DataFrame，包含整数列和日期时间列
        expected = DataFrame(
            {
                "A": [1, 2, 3, 4, 4],
                "date": pd.DatetimeIndex(
                    [
                        "2010-01-01 09:00:00",
                        "2010-01-01 09:00:01",
                        "2010-01-01 09:00:02",
                        "2010-01-01 09:00:03",
                        "NaT",  # NaT 表示不适用的日期时间
                    ]
                ),
            }
        )
        # 对预期的 DataFrame 按照列 'A' 和 'date' 进行排序
        result = expected.sort_values(["A", "date"])
        # 使用测试工具检查排序后的结果与预期的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试项目缓存的行为
    def test_sort_values_item_cache(self):
        # 之前的行为不正确，保留了一个无效的 _item_cache 条目
        # 创建一个包含随机数据的 DataFrame，并添加一个新列 'D' 等于 'A' 列的两倍
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 3)), columns=["A", "B", "C"]
        )
        ser = df["A"]
        # 断言 DataFrame 的内部块数为 2
        assert len(df._mgr.blocks) == 2

        # 对 DataFrame 按照列 'A' 进行排序，注意这里未指定 inplace=True
        df.sort_values(by="A")

        # 修改序列 ser 的第一个元素为 99
        ser.iloc[0] = 99
        # 断言 DataFrame 中第一行第一列的值没有变为 99
        assert df.iloc[0, 0] == df["A"][0]
        assert df.iloc[0, 0] != 99

    # 定义一个测试方法，用于测试数据框重塑（reshaping）的行为
    def test_sort_values_reshaping(self):
        # GH 39426: Issue reference for tracking purposes
        # 创建一个包含一行数据的 DataFrame，并按照索引排序列
        values = list(range(21))
        expected = DataFrame([values], columns=values)
        df = expected.sort_values(expected.index[0], axis=1, ignore_index=True)

        # 使用测试工具检查排序后的结果与预期的 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    # 定义一个测试方法，用于测试在 inplace=True 情况下没有指定 by 参数的行为
    def test_sort_values_no_by_inplace(self):
        # GH#50643: Issue reference for tracking purposes
        # 创建一个简单的 DataFrame
        df = DataFrame({"a": [1, 2, 3]})
        expected = df.copy()
        # 在 inplace=True 情况下对 DataFrame 按照空列表排序
        result = df.sort_values(by=[], inplace=True)
        # 使用测试工具检查排序后的结果与预期的 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)
        # 断言排序操作返回 None
        assert result is None

    # 定义一个测试方法，用于测试在 ignore_index=True 情况下的排序行为
    def test_sort_values_no_op_reset_index(self):
        # GH#52553: Issue reference for tracking purposes
        # 创建一个具有自定义索引的 DataFrame
        df = DataFrame({"A": [10, 20], "B": [1, 5]}, index=[2, 3])
        # 按照列 'A' 对 DataFrame 进行排序，并重置索引，忽略原索引
        result = df.sort_values(by="A", ignore_index=True)
        expected = DataFrame({"A": [10, 20], "B": [1, 5]})
        # 使用测试工具检查排序和重置索引后的结果与预期的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
class TestDataFrameSortKey:  # test key sorting (issue 27237)
    def test_sort_values_inplace_key(self, sort_by_key):
        # 创建一个包含随机标准正态分布数据的 DataFrame，形状为 (4, 4)，指定索引和列名
        frame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[1, 2, 3, 4],
            columns=["A", "B", "C", "D"],
        )

        # 复制 DataFrame 以避免在原地排序
        sorted_df = frame.copy()
        # 使用 sort_values 方法原地按列 "A" 排序，使用自定义的排序函数 sort_by_key
        return_value = sorted_df.sort_values(by="A", inplace=True, key=sort_by_key)
        # 断言排序方法返回 None
        assert return_value is None
        # 创建预期的排序结果
        expected = frame.sort_values(by="A", key=sort_by_key)
        # 断言排序后的 DataFrame 与预期结果相等
        tm.assert_frame_equal(sorted_df, expected)

        # 同样的过程，但是按行索引排序
        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(
            by=1, axis=1, inplace=True, key=sort_by_key
        )
        assert return_value is None
        expected = frame.sort_values(by=1, axis=1, key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)

        # 降序排序
        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(
            by="A", ascending=False, inplace=True, key=sort_by_key
        )
        assert return_value is None
        expected = frame.sort_values(by="A", ascending=False, key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)

        # 按多列排序，原地排序
        sorted_df = frame.copy()
        sorted_df.sort_values(
            by=["A", "B"], ascending=False, inplace=True, key=sort_by_key
        )
        expected = frame.sort_values(by=["A", "B"], ascending=False, key=sort_by_key)
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_key(self):
        # 创建一个包含数值和 NaN 的 DataFrame
        df = DataFrame(np.array([0, 5, np.nan, 3, 2, np.nan]))

        # 按第一列排序，NaN 在最后
        result = df.sort_values(0)
        expected = df.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_frame_equal(result, expected)

        # 按第一列排序，使用自定义排序函数（加 5）
        result = df.sort_values(0, key=lambda x: x + 5)
        expected = df.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_frame_equal(result, expected)

        # 按第一列排序，降序排序
        result = df.sort_values(0, key=lambda x: -x, ascending=False)
        expected = df.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_frame_equal(result, expected)

    def test_sort_values_by_key(self):
        # 创建一个包含两列数值和 NaN 的 DataFrame
        df = DataFrame(
            {
                "a": np.array([0, 3, np.nan, 3, 2, np.nan]),
                "b": np.array([0, 2, np.nan, 5, 2, np.nan]),
            }
        )

        # 按列 "a" 排序，使用自定义排序函数（负值）
        result = df.sort_values("a", key=lambda x: -x)
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)

        # 按多列排序，使用自定义排序函数（负值）
        result = df.sort_values(by=["a", "b"], key=lambda x: -x)
        expected = df.iloc[[3, 1, 4, 0, 2, 5]]
        tm.assert_frame_equal(result, expected)

        # 按多列排序，使用自定义排序函数（负值），降序排序
        result = df.sort_values(by=["a", "b"], key=lambda x: -x, ascending=False)
        expected = df.iloc[[0, 4, 1, 3, 2, 5]]
        tm.assert_frame_equal(result, expected)
    # 定义测试方法，验证按键名排序值的功能
    def test_sort_values_by_key_by_name(self):
        # 创建一个包含两列的 DataFrame
        df = DataFrame(
            {
                "a": np.array([0, 3, np.nan, 3, 2, np.nan]),
                "b": np.array([0, 2, np.nan, 5, 2, np.nan]),
            }
        )

        # 定义用于排序的键函数
        def key(col):
            # 如果列名是"a"，则返回其相反数
            if col.name == "a":
                return -col
            else:
                return col

        # 按照列"a"排序，使用自定义的键函数
        result = df.sort_values(by="a", key=key)
        # 预期结果是重新排列后的 DataFrame
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        # 断言排序后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 再次按照列"a"排序，使用相同的键函数
        result = df.sort_values(by=["a"], key=key)
        # 预期结果同样是重新排列后的 DataFrame
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        # 断言排序后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 按照列"b"排序，使用相同的键函数
        result = df.sort_values(by="b", key=key)
        # 预期结果是重新排列后的 DataFrame
        expected = df.iloc[[0, 1, 4, 3, 2, 5]]
        # 断言排序后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 按照列"a"和"b"同时排序，使用相同的键函数
        result = df.sort_values(by=["a", "b"], key=key)
        # 预期结果是重新排列后的 DataFrame
        expected = df.iloc[[1, 3, 4, 0, 2, 5]]
        # 断言排序后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 测试根据字符串键排序的方法
    def test_sort_values_key_string(self):
        # 创建一个包含字符串的 DataFrame
        df = DataFrame(np.array([["hello", "goodbye"], ["hello", "Hello"]]))

        # 按照列索引为1（第二列）排序
        result = df.sort_values(1)
        # 预期结果是按照第二列倒序排列的 DataFrame
        expected = df[::-1]
        # 断言排序后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 按照列索引为[0, 1]（第一列和第二列）排序，使用字符串小写化作为键函数
        result = df.sort_values([0, 1], key=lambda col: col.str.lower())
        # 预期结果是保持原顺序的 DataFrame
        tm.assert_frame_equal(result, df)

        # 按照列索引为[0, 1]（第一列和第二列）排序，使用字符串小写化作为键函数，并且降序排列
        result = df.sort_values(
            [0, 1], key=lambda col: col.str.lower(), ascending=False
        )
        # 预期结果是按照第二列倒序排列的 DataFrame
        expected = df.sort_values(1, key=lambda col: col.str.lower(), ascending=False)
        # 断言排序后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 测试空 DataFrame 的排序方法
    def test_sort_values_key_empty(self, sort_by_key):
        # 创建一个空的 DataFrame
        df = DataFrame(np.array([]))

        # 对第一列进行排序，使用给定的排序键
        df.sort_values(0, key=sort_by_key)
        # 对索引进行排序，使用给定的排序键
        df.sort_index(key=sort_by_key)

    # 测试排序时修改 DataFrame 长度会引发异常的情况
    def test_changes_length_raises(self):
        # 创建一个包含整数的 DataFrame
        df = DataFrame({"A": [1, 2, 3]})
        # 使用 pytest 断言，期望引发 ValueError 异常，并且异常信息中包含特定字符串
        with pytest.raises(ValueError, match="change the shape"):
            # 对列"A"进行排序，使用截取第一个字符的键函数
            df.sort_values("A", key=lambda x: x[:1])

    # 测试按照指定轴进行排序的方法
    def test_sort_values_key_axes(self):
        # 创建一个包含字符串和整数的 DataFrame
        df = DataFrame({0: ["Hello", "goodbye"], 1: [0, 1]})

        # 按照第一列（索引为0）排序，使用字符串小写化作为键函数
        result = df.sort_values(0, key=lambda col: col.str.lower())
        # 预期结果是按照第一列倒序排列的 DataFrame
        expected = df[::-1]
        # 断言排序后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 按照第二列（索引为1）排序，使用相反数作为键函数
        result = df.sort_values(1, key=lambda col: -col)
        # 预期结果是按照第二列倒序排列的 DataFrame
        expected = df[::-1]
        # 断言排序后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 测试按照指定字典顺序进行排序的方法
    def test_sort_values_key_dict_axis(self):
        # 创建一个包含字符串和整数的 DataFrame
        df = DataFrame({0: ["Hello", 0], 1: ["goodbye", 1]})

        # 按照第一列（索引为0）排序，使用字符串小写化作为键函数，并且指定排序的轴为列
        result = df.sort_values(0, key=lambda col: col.str.lower(), axis=1)
        # 预期结果是按照第一列倒序排列的 DataFrame
        expected = df.loc[:, ::-1]
        # 断言排序后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 按照第二列（索引为1）排序，使用相反数作为键函数，并且指定排序的轴为列
        result = df.sort_values(1, key=lambda col: -col, axis=1)
        # 预期结果是按照第二列倒序排列的 DataFrame
        expected = df.loc[:, ::-1]
        # 断言排序后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("ordered", [True, False])
    # 定义一个测试方法，用于测试排序操作，将 key 转换为分类数据类型
    def test_sort_values_key_casts_to_categorical(self, ordered):
        # GitHub 上的一个问题链接，参考：https://github.com/pandas-dev/pandas/issues/36383
        # 定义分类的顺序
        categories = ["c", "b", "a"]
        # 创建一个数据框 DataFrame，包含列 'x' 和 'y'
        df = DataFrame({"x": [1, 1, 1], "y": ["a", "b", "c"]})

        # 定义一个排序器函数 sorter，根据列名 'y' 将其转换为有序分类类型
        def sorter(key):
            if key.name == "y":
                return pd.Series(
                    Categorical(key, categories=categories, ordered=ordered)
                )
            return key

        # 对数据框 df 按照列 'x' 和 'y' 进行排序，使用自定义的排序器函数 sorter
        result = df.sort_values(by=["x", "y"], key=sorter)

        # 预期的排序结果数据框
        expected = DataFrame(
            {"x": [1, 1, 1], "y": ["c", "b", "a"]}, index=pd.Index([2, 1, 0])
        )

        # 使用测试框架中的函数比较实际结果和预期结果，确认它们相等
        tm.assert_frame_equal(result, expected)
@pytest.fixture
def df_none():
    return DataFrame(
        {
            "outer": ["a", "a", "a", "b", "b", "b"],
            "inner": [1, 2, 2, 2, 1, 1],
            "A": np.arange(6, 0, -1),
            ("B", 5): ["one", "one", "two", "two", "one", "one"],
        }
    )

@pytest.fixture(params=[["outer"], ["outer", "inner"]])
def df_idx(request, df_none):
    levels = request.param
    return df_none.set_index(levels)

@pytest.fixture(
    params=[
        "inner",  # index level
        ["outer"],  # list of index level
        "A",  # column
        [("B", 5)],  # list of column
        ["inner", "outer"],  # two index levels
        [("B", 5), "outer"],  # index level and column
        ["A", ("B", 5)],  # Two columns
        ["inner", "outer"],  # two index levels and column
    ]
)
def sort_names(request):
    return request.param

class TestSortValuesLevelAsStr:
    def test_sort_index_level_and_column_label(
        self, df_none, df_idx, sort_names, ascending, request
    ):
        # GH#14353
        # 检查是否为版本 1.25 或更高，以及测试名称是否匹配特定条件
        if (
            Version(np.__version__) >= Version("1.25")
            and request.node.callspec.id == "df_idx0-inner-True"
        ):
            # 应用标记以标记测试为预期失败，提供失败原因
            request.applymarker(
                pytest.mark.xfail(
                    reason=(
                        "pandas default unstable sorting of duplicates"
                        "issue with numpy>=1.25 with AVX instructions"
                    ),
                    strict=False,
                )
            )

        # 从 df_idx 中获取索引级别
        levels = df_idx.index.names

        # 根据指定的列排序和设置索引，计算预期结果
        expected = df_none.sort_values(
            by=sort_names, ascending=ascending, axis=0
        ).set_index(levels)

        # 根据混合的列和索引级别排序，计算实际结果
        result = df_idx.sort_values(by=sort_names, ascending=ascending, axis=0)

        # 使用测试工具比较结果和预期结果的数据框是否相等
        tm.assert_frame_equal(result, expected)

    def test_sort_column_level_and_index_label(
        self, df_none, df_idx, sort_names, ascending, request
    ):
        # GH#14353
        # 从 df_idx 中获取索引的级别（names）

        # 根据 axis=0 对 df_none 进行排序，设置索引级别，然后转置得到期望的结果。
        # 对于某些情况，这将导致具有多个列级别的数据帧
        expected = (
            df_none.sort_values(by=sort_names, ascending=ascending, axis=0)
            .set_index(levels)
            .T
        )

        # 对 df_idx 进行转置，并根据 axis=1 进行排序得到结果
        result = df_idx.T.sort_values(by=sort_names, ascending=ascending, axis=1)

        # 如果 numpy 版本 >= 1.25，则应用 pytest 的 xfail 标记，因为 pandas
        # 在 numpy>=1.25 与 AVX 指令下默认对重复值的排序不稳定。
        if Version(np.__version__) >= Version("1.25"):
            request.applymarker(
                pytest.mark.xfail(
                    reason=(
                        "pandas default unstable sorting of duplicates"
                        "issue with numpy>=1.25 with AVX instructions"
                    ),
                    strict=False,
                )
            )

        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_sort_values_validate_ascending_for_value_error(self):
        # GH41634
        # 创建一个 DataFrame，包含一个列 'D'，用于测试异常情况

        # 设置预期的错误消息内容
        msg = 'For argument "ascending" expected type bool, received type str.'
        
        # 使用 pytest.raises 检查是否会抛出 ValueError 异常，且异常消息符合预期
        with pytest.raises(ValueError, match=msg):
            df.sort_values(by="D", ascending="False")

    def test_sort_values_validate_ascending_functional(self, ascending):
        # 创建一个 DataFrame，包含一个列 'D'，用于功能性测试
        df = DataFrame({"D": [23, 7, 21]})
        
        # 获取 'D' 列的排序索引
        indexer = df["D"].argsort().values

        # 如果 ascending 不为真，则反转索引
        if not ascending:
            indexer = indexer[::-1]

        # 根据 'D' 列的排序索引重新排序 df，并设置预期结果
        expected = df.loc[df.index[indexer]]
        
        # 使用 sort_values 对 'D' 列进行排序，并获取结果
        result = df.sort_values(by="D", ascending=ascending)
        
        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
```