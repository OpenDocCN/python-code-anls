# `D:\src\scipysrc\pandas\pandas\tests\test_multilevel.py`

```
import datetime  # 导入日期时间模块

import numpy as np  # 导入NumPy模块
import pytest  # 导入pytest模块

import pandas as pd  # 导入Pandas模块
from pandas import (  # 从Pandas中导入DataFrame, MultiIndex, Series
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm  # 导入Pandas的测试工具模块


class TestMultiLevel:
    def test_reindex_level(self, multiindex_year_month_day_dataframe_random_data):
        # axis=0，获取测试数据
        ymd = multiindex_year_month_day_dataframe_random_data

        # 按照"month"分组并求和
        month_sums = ymd.groupby("month").sum()
        # 根据当前索引重新索引数据，按第二级（level=1）重新索引
        result = month_sums.reindex(ymd.index, level=1)
        # 预期结果：按"month"分组并对原数据进行求和
        expected = ymd.groupby(level="month").transform("sum")

        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)

        # 对Series进行测试
        result = month_sums["A"].reindex(ymd.index, level=1)
        expected = ymd["A"].groupby(level="month").transform("sum")
        # 断言两个Series是否相等，不检查名称
        tm.assert_series_equal(result, expected, check_names=False)

    def test_reindex(self, multiindex_dataframe_random_data):
        # 获取测试数据
        frame = multiindex_dataframe_random_data

        # 期望结果：按照指定索引重新索引DataFrame
        expected = frame.iloc[[0, 3]]
        # 根据指定索引重新索引DataFrame
        reindexed = frame.loc[[("foo", "one"), ("bar", "one")]]
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(reindexed, expected)

    def test_reindex_preserve_levels(
        self, multiindex_year_month_day_dataframe_random_data
    ):
        # 获取测试数据
        ymd = multiindex_year_month_day_dataframe_random_data

        # 创建新的索引
        new_index = ymd.index[::10]
        # 根据新索引重新索引DataFrame
        chunk = ymd.reindex(new_index)
        # 断言索引是否与新索引相同
        assert chunk.index.is_(new_index)

        # 根据新索引重新索引DataFrame
        chunk = ymd.loc[new_index]
        # 断言索引是否完全相同
        assert chunk.index.equals(new_index)

        # 转置DataFrame
        ymdT = ymd.T
        # 根据新列重新索引转置后的DataFrame
        chunk = ymdT.reindex(columns=new_index)
        # 断言列是否与新列相同
        assert chunk.columns.is_(new_index)

        # 根据新列重新索引转置后的DataFrame
        chunk = ymdT.loc[:, new_index]
        # 断言列是否完全相同
        assert chunk.columns.equals(new_index)

    def test_groupby_transform(self, multiindex_dataframe_random_data):
        # 获取测试数据
        frame = multiindex_dataframe_random_data

        # 获取"A"列数据
        s = frame["A"]
        # 获取分组器
        grouper = s.index.get_level_values(0)

        # 根据分组器进行分组，不保留分组键
        grouped = s.groupby(grouper, group_keys=False)

        # 对分组后的数据应用函数
        applied = grouped.apply(lambda x: x * 2)
        # 预期结果：对分组后的数据进行变换
        expected = grouped.transform(lambda x: x * 2)
        # 根据预期结果的索引重新索引应用后的数据
        result = applied.reindex(expected.index)
        # 断言两个Series是否相等，不检查名称
        tm.assert_series_equal(result, expected, check_names=False)

    def test_groupby_corner(self):
        # 创建多级索引
        midx = MultiIndex(
            levels=[["foo"], ["bar"], ["baz"]],
            codes=[[0], [0], [0]],
            names=["one", "two", "three"],
        )
        # 创建DataFrame
        df = DataFrame(
            [np.random.default_rng(2).random(4)],
            columns=["a", "b", "c", "d"],
            index=midx,
        )
        # 应当正常工作
        df.groupby(level="three")

    def test_setitem_with_expansion_multiindex_columns(
        self, multiindex_year_month_day_dataframe_random_data
    ):
        # 获取测试数据
        ymd = multiindex_year_month_day_dataframe_random_data

        # 转置DataFrame，并获取前5行数据
        df = ymd[:5].T
        # 将指定列设置为另一列的值
        df[2000, 1, 10] = df[2000, 1, 7]
        # 断言列是否为多级索引
        assert isinstance(df.columns, MultiIndex)
        # 断言新列的值是否与原列的值完全相同
        assert (df[2000, 1, 10] == df[2000, 1, 7]).all()
    # 测试对齐功能
    def test_alignment(self):
        # 创建第一个 Series 对象 x，指定数据和多级索引
        x = Series(
            data=[1, 2, 3], index=MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 3)])
        )

        # 创建第二个 Series 对象 y，指定数据和多级索引
        y = Series(
            data=[4, 5, 6], index=MultiIndex.from_tuples([("Z", 1), ("Z", 2), ("B", 3)])
        )

        # 计算 x 和 y 的差值
        res = x - y
        # 计算期望的索引，为 x 和 y 的索引的并集
        exp_index = x.index.union(y.index)
        # 计算期望结果，先对 x 和 y 按期望索引重建，再计算差值
        exp = x.reindex(exp_index) - y.reindex(exp_index)
        # 使用断言检查 res 是否等于 exp
        tm.assert_series_equal(res, exp)

        # 测试非单调代码路径
        # 对 x 和 y 进行逆序操作，然后再次计算差值
        res = x[::-1] - y[::-1]
        # 重新计算期望的索引
        exp_index = x.index.union(y.index)
        # 再次计算期望结果，按照新的索引重建 x 和 y，并计算差值
        exp = x.reindex(exp_index) - y.reindex(exp_index)
        # 使用断言检查逆序计算的 res 是否等于逆序计算的 exp
        tm.assert_series_equal(res, exp)

    # 测试多级分组功能
    def test_groupby_multilevel(self, multiindex_year_month_day_dataframe_random_data):
        # 获取随机生成的多级索引日期数据框
        ymd = multiindex_year_month_day_dataframe_random_data

        # 按照第一和第二级索引分组，并计算均值
        result = ymd.groupby(level=[0, 1]).mean()

        # 获取第一级索引和第二级索引的值
        k1 = ymd.index.get_level_values(0)
        k2 = ymd.index.get_level_values(1)

        # 按照 k1 和 k2 分组，并计算均值，作为期望结果
        expected = ymd.groupby([k1, k2]).mean()

        # 使用断言检查结果 result 是否等于期望结果 expected
        tm.assert_frame_equal(result, expected)
        # 使用断言检查结果的索引名是否与原始数据框的索引名的前两个相同
        assert result.index.names == ymd.index.names[:2]

        # 再次按照第一和第二级索引分组，并计算均值
        result2 = ymd.groupby(level=ymd.index.names[:2]).mean()
        # 使用断言检查之前计算的 result 与当前计算的 result2 是否相等
        tm.assert_frame_equal(result, result2)

    # 测试多级索引数据框的合并功能
    def test_multilevel_consolidate(self):
        # 创建多级索引对象 index
        index = MultiIndex.from_tuples(
            [("foo", "one"), ("foo", "two"), ("bar", "one"), ("bar", "two")]
        )
        # 创建随机数据框 df，指定索引和列名
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)), index=index, columns=index
        )
        # 添加一列总和，并返回数据框的已合并版本
        df["Totals", ""] = df.sum(axis=1)
        df = df._consolidate()
    def test_level_with_tuples(self):
        # 创建一个多级索引对象，包含两个级别：第一级是元组列表，第二级是整数列表
        index = MultiIndex(
            levels=[[("foo", "bar", 0), ("foo", "baz", 0), ("foo", "qux", 0)], [0, 1]],
            codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        )

        # 创建一个 Series 对象，使用随机生成的数据，并使用上面创建的多级索引
        series = Series(np.random.default_rng(2).standard_normal(6), index=index)
        
        # 创建一个 DataFrame 对象，使用随机生成的数据，并使用上面创建的多级索引
        frame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=index)

        # 使用多级索引进行 Series 的索引操作，返回结果
        result = series[("foo", "bar", 0)]
        
        # 使用 .loc 方法进行 Series 的索引操作，返回结果
        result2 = series.loc[("foo", "bar", 0)]
        
        # 期望结果是 series 的前两个元素组成的 Series 对象
        expected = series[:2]
        
        # 将期望结果的索引降级，去掉第一级索引
        expected.index = expected.index.droplevel(0)
        
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)

        # 使用 pytest 来断言索引操作引发 KeyError 异常，异常信息要匹配指定的正则表达式
        with pytest.raises(KeyError, match=r"^\(\('foo', 'bar', 0\), 2\)$"):
            series[("foo", "bar", 0), 2]

        # 使用 .loc 方法进行 DataFrame 的索引操作，返回结果
        result = frame.loc[("foo", "bar", 0)]
        
        # 使用 .xs 方法进行 DataFrame 的索引操作，返回结果
        result2 = frame.xs(("foo", "bar", 0))
        
        # 期望结果是 frame 的前两行
        expected = frame[:2]
        
        # 将期望结果的索引降级，去掉第一级索引
        expected.index = expected.index.droplevel(0)
        
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        # 创建一个新的多级索引对象，包含两个级别：第一级是元组列表，第二级是整数列表
        index = MultiIndex(
            levels=[[("foo", "bar"), ("foo", "baz"), ("foo", "qux")], [0, 1]],
            codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        )

        # 使用新的多级索引创建 Series 对象，使用随机生成的数据
        series = Series(np.random.default_rng(2).standard_normal(6), index=index)
        
        # 使用新的多级索引创建 DataFrame 对象，使用随机生成的数据
        frame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=index)

        # 使用多级索引进行 Series 的索引操作，返回结果
        result = series[("foo", "bar")]
        
        # 使用 .loc 方法进行 Series 的索引操作，返回结果
        result2 = series.loc[("foo", "bar")]
        
        # 期望结果是 series 的前两个元素组成的 Series 对象
        expected = series[:2]
        
        # 将期望结果的索引降级，去掉第一级索引
        expected.index = expected.index.droplevel(0)
        
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)

        # 使用 .loc 方法进行 DataFrame 的索引操作，返回结果
        result = frame.loc[("foo", "bar")]
        
        # 使用 .xs 方法进行 DataFrame 的索引操作，返回结果
        result2 = frame.xs(("foo", "bar"))
        
        # 期望结果是 frame 的前两行
        expected = frame[:2]
        
        # 将期望结果的索引降级，去掉第一级索引
        expected.index = expected.index.droplevel(0)
        
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

    def test_reindex_level_partial_selection(self, multiindex_dataframe_random_data):
        # 从测试夹具中获取多级索引的 DataFrame 对象
        frame = multiindex_dataframe_random_data

        # 使用 reindex 方法，按照第一个级别的指定索引重新索引 DataFrame
        result = frame.reindex(["foo", "qux"], level=0)
        
        # 期望结果是原 DataFrame 的指定行和对应的重新索引结果
        expected = frame.iloc[[0, 1, 2, 7, 8, 9]]
        
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)

        # 对 DataFrame 的转置结果使用 reindex 方法，按照第一个级别的指定索引重新索引
        result = frame.T.reindex(["foo", "qux"], axis=1, level=0)
        
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected.T)

        # 使用 .loc 方法按照指定索引选择 DataFrame 的行
        result = frame.loc[["foo", "qux"]]
        
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)

        # 使用 [] 运算符选择 DataFrame 的列，并按照指定索引选择其中的行
        result = frame["A"].loc[["foo", "qux"]]
        
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected["A"])

        # 对 DataFrame 的转置结果使用 .loc 方法，按照指定的索引选择其中的列
        result = frame.T.loc[:, ["foo", "qux"]]
        
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected.T)

    @pytest.mark.parametrize("d", [4, "d"])
    def test_empty_frame_groupby_dtypes_consistency(self, d):
        # GH 20888
        # 定义分组键
        group_keys = ["a", "b", "c"]
        # 创建一个DataFrame对象，包含列名和数据
        df = DataFrame({"a": [1], "b": [2], "c": [3], "d": [d]})

        # 根据条件过滤DataFrame并进行分组
        g = df[df.a == 2].groupby(group_keys)
        # 获取分组后每组的第一个索引值
        result = g.first().index
        # 创建一个预期的多级索引对象
        expected = MultiIndex(
            levels=[[1], [2], [3]], codes=[[], [], []], names=["a", "b", "c"]
        )

        # 断言实际结果与预期结果相等
        tm.assert_index_equal(result, expected)

    def test_duplicate_groupby_issues(self):
        # 创建包含元组的列表
        idx_tp = [
            ("600809", "20061231"),
            ("600809", "20070331"),
            ("600809", "20070630"),
            ("600809", "20070331"),
        ]
        # 创建与元组列表对应的数据列表
        dt = ["demo", "demo", "demo", "demo"]

        # 使用元组列表创建多级索引
        idx = MultiIndex.from_tuples(idx_tp, names=["STK_ID", "RPT_Date"])
        # 使用数据列表和索引创建Series对象
        s = Series(dt, index=idx)

        # 按照索引分组并获取每组的第一个值
        result = s.groupby(s.index).first()
        # 断言结果长度为3
        assert len(result) == 3

    def test_subsets_multiindex_dtype(self):
        # GH 20757
        # 创建包含数据和列索引的列表
        data = [["x", 1]]
        columns = [("a", "b", np.nan), ("a", "c", 0.0)]
        # 使用数据和列索引列表创建DataFrame对象
        df = DataFrame(data, columns=MultiIndex.from_tuples(columns))
        # 获取预期结果的数据类型
        expected = df.dtypes.a.b
        # 获取实际结果的数据类型
        result = df.a.b.dtypes
        # 断言Series对象的数据类型相等
        tm.assert_series_equal(result, expected)

    def test_datetime_object_multiindex(self):
        # 创建包含元组键和对应字典的数据字典
        data_dic = {
            (0, datetime.date(2018, 3, 3)): {"A": 1, "B": 10},
            (0, datetime.date(2018, 3, 4)): {"A": 2, "B": 11},
            (1, datetime.date(2018, 3, 3)): {"A": 3, "B": 12},
            (1, datetime.date(2018, 3, 4)): {"A": 4, "B": 13},
        }
        # 使用数据字典创建DataFrame对象
        result = DataFrame.from_dict(data_dic, orient="index")
        # 创建包含数据和索引的数据字典
        data = {"A": [1, 2, 3, 4], "B": [10, 11, 12, 13]}
        index = [
            [0, 0, 1, 1],
            [
                datetime.date(2018, 3, 3),
                datetime.date(2018, 3, 4),
                datetime.date(2018, 3, 3),
                datetime.date(2018, 3, 4),
            ],
        ]
        # 使用数据和索引数据字典创建DataFrame对象
        expected = DataFrame(data=data, index=index)

        # 断言DataFrame对象相等
        tm.assert_frame_equal(result, expected)

    def test_multiindex_with_na(self):
        # 创建包含列表的DataFrame对象，其中包含NaN值
        df = DataFrame(
            [
                ["A", np.nan, 1.23, 4.56],
                ["A", "G", 1.23, 4.56],
                ["A", "D", 9.87, 10.54],
            ],
            columns=["pivot_0", "pivot_1", "col_1", "col_2"],
        ).set_index(["pivot_0", "pivot_1"])

        # 在DataFrame中特定位置设置值为0.0
        df.at[("A", "F"), "col_2"] = 0.0

        # 创建包含列表的预期DataFrame对象，其中包含NaN值
        expected = DataFrame(
            [
                ["A", np.nan, 1.23, 4.56],
                ["A", "G", 1.23, 4.56],
                ["A", "D", 9.87, 10.54],
                ["A", "F", np.nan, 0.0],
            ],
            columns=["pivot_0", "pivot_1", "col_1", "col_2"],
        ).set_index(["pivot_0", "pivot_1"])

        # 断言DataFrame对象相等
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("na", [None, np.nan])
    # 定义一个测试方法，用于测试在多重索引中插入带有缺失值的情况
    def test_multiindex_insert_level_with_na(self, na):
        # GH 59003：参考 GitHub issue 59003
        # 创建一个 DataFrame，包含一个元素为 0 的列，列名为多重索引 ["A", "B"]
        df = DataFrame([0], columns=[["A"], ["B"]])
        # 在指定的多重索引位置 [na, "B"] 处插入值为 1
        df[na, "B"] = 1
        # 使用测试框架检查 df[na] 是否与期望的 DataFrame([1], columns=["B"]) 相等
        tm.assert_frame_equal(df[na], DataFrame([1], columns=["B"]))
class TestSorted:
    """everything you wanted to test about sorting"""

    def test_sort_non_lexsorted(self):
        # 创建一个 MultiIndex 对象，用于测试排序
        # 包含了两个级别的索引和对应的数据
        # GH 15797
        idx = MultiIndex(
            [["A", "B", "C"], ["c", "b", "a"]], [[0, 1, 2, 0, 1, 2], [0, 2, 1, 1, 0, 2]]
        )

        # 创建一个 DataFrame 对象，使用上述的 MultiIndex 作为索引
        # 数据列为从 0 开始的递增整数，数据类型为 int64
        df = DataFrame({"col": range(len(idx))}, index=idx, dtype="int64")
        # 断言索引是否是单调递增的，应该为 False
        assert df.index.is_monotonic_increasing is False

        # 对 DataFrame 进行排序，并重新赋值给 sorted 变量
        sorted = df.sort_index()
        # 断言排序后的索引是否是单调递增的，应该为 True
        assert sorted.index.is_monotonic_increasing is True

        # 创建一个预期的 DataFrame 对象，包含特定的数据和 MultiIndex 索引
        expected = DataFrame(
            {"col": [1, 4, 5, 2]},
            index=MultiIndex.from_tuples(
                [("B", "a"), ("B", "c"), ("C", "a"), ("C", "b")]
            ),
            dtype="int64",
        )
        # 使用 loc 方法从 sorted DataFrame 中选择子集，赋值给 result 变量
        result = sorted.loc[pd.IndexSlice["B":"C", "a":"c"], :]
        # 使用 assert_frame_equal 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
```