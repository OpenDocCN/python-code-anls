# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_grouping.py`

```
"""
test where we are determining what we are grouping, or getting groups
"""

from datetime import (
    date,
    timedelta,
)

import numpy as np
import pytest

from pandas.errors import SpecificationError

import pandas as pd
from pandas import (
    CategoricalIndex,
    DataFrame,
    Grouper,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping

# selection
# --------------------------------


class TestSelection:
    # 测试选择不正确的列时是否会引发 KeyError 异常，异常消息应包含指定的列名
    def test_select_bad_cols(self):
        df = DataFrame([[1, 2]], columns=["A", "B"])
        g = df.groupby("A")
        with pytest.raises(KeyError, match="\"Columns not found: 'C'\""):
            g[["C"]]

        with pytest.raises(KeyError, match="^[^A]+$"):
            # A should not be referenced as a bad column...
            # will have to rethink regex if you change message!
            g[["A", "C"]]

    # 测试当存在重复列名时是否会引发 ValueError 异常，异常消息应指出 Grouper 需要是一维的
    def test_groupby_duplicated_column_errormsg(self):
        # GH7511
        df = DataFrame(
            columns=["A", "B", "A", "C"], data=[range(4), range(2, 6), range(0, 8, 2)]
        )

        msg = "Grouper for 'A' not 1-dimensional"
        with pytest.raises(ValueError, match=msg):
            df.groupby("A")
        with pytest.raises(ValueError, match=msg):
            df.groupby(["A", "B"])

        grouped = df.groupby("B")
        c = grouped.count()
        assert c.columns.nlevels == 1
        assert c.columns.size == 3

    # 测试通过属性选择列时的结果是否与预期相符
    def test_column_select_via_attr(self, df):
        result = df.groupby("A").C.sum()
        expected = df.groupby("A")["C"].sum()
        tm.assert_series_equal(result, expected)

        df["mean"] = 1.5
        result = df.groupby("A").mean(numeric_only=True)
        expected = df.groupby("A")[["C", "D", "mean"]].agg("mean")
        tm.assert_frame_equal(result, expected)

    # 测试通过列表选择列的结果是否与预期相符
    def test_getitem_list_of_columns(self):
        df = DataFrame(
            {
                "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
                "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
                "C": np.random.default_rng(2).standard_normal(8),
                "D": np.random.default_rng(2).standard_normal(8),
                "E": np.random.default_rng(2).standard_normal(8),
            }
        )

        result = df.groupby("A")[["C", "D"]].mean()
        result2 = df.groupby("A")[df.columns[2:4]].mean()

        expected = df.loc[:, ["A", "C", "D"]].groupby("A").mean()

        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)
    # 定义一个测试函数，用于测试 DataFrame 对象中通过数值列名获取分组后均值的功能
    def test_getitem_numeric_column_names(self):
        # 创建一个 DataFrame 对象，包含数值列和随机生成的数据
        df = DataFrame(
            {
                0: list("abcd") * 2,
                2: np.random.default_rng(2).standard_normal(8),
                4: np.random.default_rng(2).standard_normal(8),
                6: np.random.default_rng(2).standard_normal(8),
            }
        )
        # 使用 DataFrame 的 groupby 方法对列 0 进行分组，并计算列 1 和列 2 的均值
        result = df.groupby(0)[df.columns[1:3]].mean()
        # 使用 DataFrame 的 groupby 方法对列 0 进行分组，并计算列 2 和列 4 的均值
        result2 = df.groupby(0)[[2, 4]].mean()

        # 期望的结果是从 DataFrame 中选择列 0、2、4，然后对列 0 进行分组后求均值
        expected = df.loc[:, [0, 2, 4]].groupby(0).mean()

        # 使用 pytest 中的 assert_frame_equal 函数比较 result 和 expected，确保它们相等
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        # 根据 GH 23566 强制废弃的规定，使用 pytest 的 raises 函数确保在尝试以元组形式选取列时会抛出 ValueError 异常
        with pytest.raises(ValueError, match="Cannot subset columns with a tuple"):
            df.groupby(0)[2, 4].mean()

    # 定义一个测试函数，测试在使用单个元组列名时是否会抛出 ValueError 异常
    def test_getitem_single_tuple_of_columns_raises(self, df):
        # 根据 GH 23566 强制废弃的规定，使用 pytest 的 raises 函数确保在尝试以元组形式选取列时会抛出 ValueError 异常
        with pytest.raises(ValueError, match="Cannot subset columns with a tuple"):
            df.groupby("A")["C", "D"].mean()

    # 定义一个测试函数，测试在使用单个列名时能否正确计算分组后的均值
    def test_getitem_single_column(self):
        # 创建一个包含字符串和随机生成数据的 DataFrame 对象
        df = DataFrame(
            {
                "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
                "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
                "C": np.random.default_rng(2).standard_normal(8),
                "D": np.random.default_rng(2).standard_normal(8),
                "E": np.random.default_rng(2).standard_normal(8),
            }
        )

        # 使用 DataFrame 的 groupby 方法对列 A 进行分组，并计算列 C 的均值
        result = df.groupby("A")["C"].mean()

        # 将 DataFrame 按列 A 和 C 进行选择并进行分组，然后计算均值
        as_frame = df.loc[:, ["A", "C"]].groupby("A").mean()
        # 从 as_frame 中选取第一列作为 Series 对象
        as_series = as_frame.iloc[:, 0]
        # 期望的结果是 as_series
        expected = as_series

        # 使用 pytest 的 assert_series_equal 函数比较 result 和 expected，确保它们相等
        tm.assert_series_equal(result, expected)

    # 使用 pytest 的参数化标记，定义一个测试函数，测试从 grouper 中选取元素并应用函数 func 的功能
    @pytest.mark.parametrize(
        "func", [lambda x: x.sum(), lambda x: x.agg(lambda y: y.sum())]
    )
    def test_getitem_from_grouper(self, func):
        # 创建一个包含数值和字母的 DataFrame 对象
        df = DataFrame({"a": [1, 1, 2], "b": 3, "c": 4, "d": 5})
        # 使用 groupby 方法将列 a 和 b 分组，并选择列 a 和 c
        gb = df.groupby(["a", "b"])[["a", "c"]]

        # 创建一个 MultiIndex 对象，用于标识分组后的结果
        idx = MultiIndex.from_tuples([(1, 3), (2, 3)], names=["a", "b"])
        # 创建一个期望的 DataFrame，包含从 gb 中选择列 a 和 c 后求和的结果
        expected = DataFrame({"a": [2, 2], "c": [8, 4]}, index=idx)
        # 使用 func 函数对 gb 进行计算
        result = func(gb)

        # 使用 pytest 的 assert_frame_equal 函数比较 result 和 expected，确保它们相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，测试在使用 lambda 表达式作为 grouper 时是否能正确返回分组的 indices
    def test_indices_grouped_by_tuple_with_lambda(self):
        # 创建一个包含元组的 DataFrame 对象
        df = DataFrame(
            {
                "Tuples": (
                    (x, y)
                    for x in [0, 1]
                    for y in np.random.default_rng(2).integers(3, 5, 5)
                )
            }
        )

        # 使用 groupby 方法按 Tuples 列分组，并返回分组后的 indices
        gb = df.groupby("Tuples")
        # 使用 lambda 表达式对 df 进行分组，并返回分组后的 indices
        gb_lambda = df.groupby(lambda x: df.iloc[x, 0])

        # 期望的结果是 gb 的 indices
        expected = gb.indices
        # 实际的结果是 gb_lambda 的 indices
        result = gb_lambda.indices

        # 使用 pytest 的 assert_dict_equal 函数比较 result 和 expected，确保它们相等
        tm.assert_dict_equal(result, expected)
# grouping
# --------------------------------

class TestGrouping:
    @pytest.mark.parametrize(
        "index",
        [
            # 创建一个 Index 对象，使用字符列表作为索引
            Index(list("abcde")),
            # 创建一个 Index 对象，使用 numpy 数组作为索引
            Index(np.arange(5)),
            # 创建一个 Index 对象，使用浮点数 numpy 数组作为索引
            Index(np.arange(5, dtype=float)),
            # 创建一个日期范围对象，从 "2020-01-01" 开始，连续5天
            date_range("2020-01-01", periods=5),
            # 创建一个周期范围对象，从 "2020-01-01" 开始，连续5个周期
            period_range("2020-01-01", periods=5),
        ],
    )
    def test_grouper_index_types(self, index):
        # related GH5375
        # 在使用 Floatlike 索引时，groupby 函数的异常行为
        # 创建一个 DataFrame 对象，使用 numpy 数组作为数据，指定列名和给定的索引对象
        df = DataFrame(np.arange(10).reshape(5, 2), columns=list("AB"), index=index)

        # 对 DataFrame 进行按列分组，并应用 lambda 函数
        df.groupby(list("abcde"), group_keys=False).apply(lambda x: x)

        # 翻转 DataFrame 的索引
        df.index = df.index[::-1]
        # 对 DataFrame 进行按列分组，并应用 lambda 函数
        df.groupby(list("abcde"), group_keys=False).apply(lambda x: x)

    def test_grouper_multilevel_freq(self):
        # GH 7885
        # 在 Grouper 中指定 level 和 freq 参数
        d0 = date.today() - timedelta(days=14)
        dates = date_range(d0, date.today())
        # 创建一个多级索引对象，使用日期列表的笛卡尔积
        date_index = MultiIndex.from_product([dates, dates], names=["foo", "bar"])
        # 创建一个 DataFrame 对象，使用随机整数填充，指定多级索引
        df = DataFrame(np.random.default_rng(2).integers(0, 100, 225), index=date_index)

        # 检查字符串级别的分组
        expected = (
            df.reset_index()
            .groupby([Grouper(key="foo", freq="W"), Grouper(key="bar", freq="W")])
            .sum()
        )
        # 重置索引会导致列的数据类型变为对象
        expected.columns = Index([0], dtype="int64")

        # 按照指定的级别和频率进行分组，并对数据进行求和
        result = df.groupby(
            [Grouper(level="foo", freq="W"), Grouper(level="bar", freq="W")]
        ).sum()
        tm.assert_frame_equal(result, expected)

        # 检查整数级别的分组
        result = df.groupby(
            [Grouper(level=0, freq="W"), Grouper(level=1, freq="W")]
        ).sum()
        tm.assert_frame_equal(result, expected)

    def test_grouper_creation_bug(self):
        # GH 8795
        # 测试 Grouper 对象的创建 bug
        df = DataFrame({"A": [0, 0, 1, 1, 2, 2], "B": [1, 2, 3, 4, 5, 6]})
        # 按列 'A' 进行分组
        g = df.groupby("A")
        expected = g.sum()

        # 使用 Grouper 对象按照 'A' 列进行分组
        g = df.groupby(Grouper(key="A"))
        result = g.sum()
        tm.assert_frame_equal(result, expected)

        # 测试使用 apply 函数时产生的警告信息
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            result = g.apply(lambda x: x.sum())
        # 从期望结果中选择特定的列
        expected["A"] = [0, 2, 4]
        expected = expected.loc[:, ["A", "B"]]
        tm.assert_frame_equal(result, expected)
    def test_grouper_creation_bug2(self):
        # GH14334
        # Grouper(key=...) may be passed in a list
        # 创建一个包含列 A、B、C 的数据帧 DataFrame
        df = DataFrame(
            {"A": [0, 0, 0, 1, 1, 1], "B": [1, 1, 2, 2, 3, 3], "C": [1, 2, 3, 4, 5, 6]}
        )
        
        # 按单列 A 进行分组
        expected = df.groupby("A").sum()
        g = df.groupby([Grouper(key="A")])
        result = g.sum()
        # 检验结果是否符合预期
        tm.assert_frame_equal(result, expected)

        # 按两列 A 和 B 进行分组
        expected = df.groupby(["A", "B"]).sum()
        
        # 使用字符串和 Grouper 对象的组合进行分组
        g = df.groupby([Grouper(key="A"), Grouper(key="B")])
        result = g.sum()
        # 检验结果是否符合预期
        tm.assert_frame_equal(result, expected)

        # 使用字符串和 Grouper 对象的组合进行分组
        g = df.groupby(["A", Grouper(key="B")])
        result = g.sum()
        # 检验结果是否符合预期
        tm.assert_frame_equal(result, expected)

        # 使用 Grouper 对象和字符串的组合进行分组
        g = df.groupby([Grouper(key="A"), "B"])
        result = g.sum()
        # 检验结果是否符合预期
        tm.assert_frame_equal(result, expected)

    def test_grouper_creation_bug3(self, unit):
        # GH8866
        # 创建一个时间范围为两天的日期索引对象 dti
        dti = date_range("20130101", periods=2, unit=unit)
        # 创建一个多重索引对象 mi，包含三个级别 one、two、three
        mi = MultiIndex.from_product(
            [list("ab"), range(2), dti],
            names=["one", "two", "three"],
        )
        # 创建一个序列对象 ser，包含从0到7的整数，使用 mi 作为索引
        ser = Series(
            np.arange(8, dtype="int64"),
            index=mi,
        )
        # 使用 Grouper 对象按 level="three" 和 freq="ME" 进行分组，求和
        result = ser.groupby(Grouper(level="three", freq="ME")).sum()
        # 创建预期的日期索引对象 exp_dti，频率为 "ME"
        exp_dti = pd.DatetimeIndex(
            [Timestamp("2013-01-31")], freq="ME", name="three"
        ).as_unit(unit)
        # 创建预期的序列对象 expected，包含和为28
        expected = Series(
            [28],
            index=exp_dti,
        )
        # 检验结果是否符合预期
        tm.assert_series_equal(result, expected)

        # 只指定一个 level="one" 会导致错误
        result = ser.groupby(Grouper(level="one")).sum()
        expected = ser.groupby(level="one").sum()
        # 检验结果是否符合预期
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("func", [False, True])
    def test_grouper_returning_tuples(self, func):
        # GH 22257 , both with dict and with callable
        # 创建一个包含列 X、Y 的数据帧 DataFrame
        df = DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]})
        # 创建一个字典 mapping，将范围内的索引映射到元组
        mapping = dict(zip(range(4), [("C", 5), ("D", 6)] * 2))

        # 根据 func 的值选择使用字典或者可调用对象进行分组
        if func:
            gb = df.groupby(by=lambda idx: mapping[idx], sort=False)
        else:
            gb = df.groupby(by=mapping, sort=False)

        # 获取第一个分组的名称和对应的数据帧 expected
        name, expected = next(iter(gb))
        assert name == ("C", 5)
        result = gb.get_group(name)

        # 检验结果是否符合预期
        tm.assert_frame_equal(result, expected)
    def test_grouper_column_and_index(self):
        # GH 14327

        # Grouping a multi-index frame by a column and an index level should
        # be equivalent to resetting the index and grouping by two columns
        # 创建一个多级索引的数据框 df_multi
        idx = MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("a", 3), ("b", 1), ("b", 2), ("b", 3)]
        )
        idx.names = ["outer", "inner"]
        df_multi = DataFrame(
            {"A": np.arange(6), "B": ["one", "one", "two", "two", "one", "one"]},
            index=idx,
        )
        # 对 df_multi 按 ["B", "inner"] 分组求均值，numeric_only=True 表示仅对数值列求均值
        result = df_multi.groupby(["B", Grouper(level="inner")]).mean(numeric_only=True)
        # 将 df_multi 重置索引后，按 ["B", "inner"] 分组求均值，numeric_only=True 表示仅对数值列求均值
        expected = (
            df_multi.reset_index().groupby(["B", "inner"]).mean(numeric_only=True)
        )
        # 断言两个结果是否相等
        tm.assert_frame_equal(result, expected)

        # Test the reverse grouping order
        # 测试反向分组顺序
        result = df_multi.groupby([Grouper(level="inner"), "B"]).mean(numeric_only=True)
        # 将 df_multi 重置索引后，按 ["inner", "B"] 分组求均值，numeric_only=True 表示仅对数值列求均值
        expected = (
            df_multi.reset_index().groupby(["inner", "B"]).mean(numeric_only=True)
        )
        # 断言两个结果是否相等
        tm.assert_frame_equal(result, expected)

        # Grouping a single-index frame by a column and the index should
        # be equivalent to resetting the index and grouping by two columns
        # 创建一个单级索引的数据框 df_single，根据 "outer" 列重置索引
        df_single = df_multi.reset_index("outer")
        # 对 df_single 按 ["B", "inner"] 分组求均值，numeric_only=True 表示仅对数值列求均值
        result = df_single.groupby(["B", Grouper(level="inner")]).mean(
            numeric_only=True
        )
        # 将 df_single 重置索引后，按 ["B", "inner"] 分组求均值，numeric_only=True 表示仅对数值列求均值
        expected = (
            df_single.reset_index().groupby(["B", "inner"]).mean(numeric_only=True)
        )
        # 断言两个结果是否相等
        tm.assert_frame_equal(result, expected)

        # Test the reverse grouping order
        # 测试反向分组顺序
        result = df_single.groupby([Grouper(level="inner"), "B"]).mean(
            numeric_only=True
        )
        # 将 df_single 重置索引后，按 ["inner", "B"] 分组求均值，numeric_only=True 表示仅对数值列求均值
        expected = (
            df_single.reset_index().groupby(["inner", "B"]).mean(numeric_only=True)
        )
        # 断言两个结果是否相等
        tm.assert_frame_equal(result, expected)

    def test_groupby_levels_and_columns(self):
        # GH9344, GH9049
        # 创建多级索引名列表 idx_names
        idx_names = ["x", "y"]
        # 用给定的多级索引创建数据框 df
        idx = MultiIndex.from_tuples([(1, 1), (1, 2), (3, 4), (5, 6)], names=idx_names)
        df = DataFrame(np.arange(12).reshape(-1, 3), index=idx)

        # 按 idx_names 指定的多级索引级别分组求均值
        by_levels = df.groupby(level=idx_names).mean()
        # 重置索引后，按 idx_names 指定的列分组求均值
        by_columns = df.reset_index().groupby(idx_names).mean()

        # 将 by_columns 的列数据类型转换为 np.int64
        by_columns.columns = by_columns.columns.astype(np.int64)
        # 断言两个结果是否相等
        tm.assert_frame_equal(by_levels, by_columns)
    def test_groupby_categorical_index_and_columns(self, observed):
        # GH18432, adapted for GH25871
        # 定义列名
        columns = ["A", "B", "A", "B"]
        # 定义分类顺序
        categories = ["B", "A"]
        # 创建数据数组
        data = np.array(
            [[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2]], int
        )
        # 使用自定义分类索引创建分类索引对象
        cat_columns = CategoricalIndex(columns, categories=categories, ordered=True)
        # 创建预期的数据数组
        expected_data = np.array([[4, 2], [4, 2], [4, 2], [4, 2], [4, 2]], int)
        # 使用自定义分类索引创建预期的列名索引对象
        expected_columns = CategoricalIndex(
            categories, categories=categories, ordered=True
        )

        # 测试转置版本的数据框
        df = DataFrame(data.T, index=cat_columns)
        # 对数据框进行分组，并求和
        result = df.groupby(level=0, observed=observed).sum()
        # 创建预期的数据框
        expected = DataFrame(data=expected_data.T, index=expected_columns)
        # 断言结果与预期相等
        tm.assert_frame_equal(result, expected)

    def test_grouper_getting_correct_binner(self):
        # GH 10063
        # 使用非时间基准的分组器和时间基准的分组器，并指定级别
        # 创建具有多重索引的数据框
        df = DataFrame(
            {"A": 1},
            index=MultiIndex.from_product(
                [list("ab"), date_range("20130101", periods=80)], names=["one", "two"]
            ),
        )
        # 对数据框进行分组，并求和
        result = df.groupby(
            [Grouper(level="one"), Grouper(level="two", freq="ME")]
        ).sum()
        # 创建预期的数据框
        expected = DataFrame(
            {"A": [31, 28, 21, 31, 28, 21]},
            index=MultiIndex.from_product(
                [list("ab"), date_range("20130101", freq="ME", periods=3)],
                names=["one", "two"],
            ),
        )
        # 断言结果与预期相等
        tm.assert_frame_equal(result, expected)

    def test_grouper_iter(self, df):
        # 对数据框进行分组，并获取分组器
        gb = df.groupby("A")
        grouper = gb._grouper
        # 对分组器的结果进行排序
        result = sorted(grouper)
        expected = ["bar", "foo"]
        # 断言排序后的结果与预期相等
        assert result == expected

    def test_empty_groups(self, df):
        # 见 gh-1048
        # 使用 pytest 断言，验证抛出 ValueError 异常，匹配给定的错误消息
        with pytest.raises(ValueError, match="No group keys passed!"):
            df.groupby([])

    def test_groupby_grouper(self, df):
        # 对数据框进行分组
        grouped = df.groupby("A")
        grouper = grouped._grouper
        # 对分组器进行分组并计算平均值（仅数值列）
        result = df.groupby(grouper).mean(numeric_only=True)
        expected = grouped.mean(numeric_only=True)
        # 断言结果与预期相等
        tm.assert_frame_equal(result, expected)
    # 定义测试方法，验证 groupby 函数在使用字典作为参数时的行为
    def test_groupby_dict_mapping(self):
        # 创建包含单个键值对的 Series 对象
        s = Series({"T1": 5})
        # 使用字典 {"T1": "T2"} 对 Series 进行分组，并对分组后的结果求和
        result = s.groupby({"T1": "T2"}).agg("sum")
        # 创建预期的 Series 对象，使用键 "T2" 对原始 Series 进行分组并求和
        expected = s.groupby(["T2"]).agg("sum")
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

        # 创建包含四个元素的 Series 对象，指定索引为 ['a', 'b', 'c', 'd']
        s = Series([1.0, 2.0, 3.0, 4.0], index=list("abcd"))
        # 创建一个映射关系，将索引映射到分组的键
        mapping = {"a": 0, "b": 0, "c": 1, "d": 1}

        # 使用映射对 Series 进行分组，并计算每组的均值
        result = s.groupby(mapping).mean()
        # 使用映射对 Series 进行分组，并对每组数据进行均值计算
        result2 = s.groupby(mapping).agg("mean")
        # 创建预期的分组键，以便对原始 Series 进行分组和均值计算
        exp_key = np.array([0, 0, 1, 1], dtype=np.int64)
        # 使用预期的分组键对原始 Series 进行分组，并计算每组的均值
        expected = s.groupby(exp_key).mean()
        # 使用预期的分组键对原始 Series 进行分组，并计算每组的均值
        expected2 = s.groupby(exp_key).mean()
        # 断言结果与预期均值 Series 对象相等
        tm.assert_series_equal(result, expected)
        # 断言结果与第二个预期均值 Series 对象相等
        tm.assert_series_equal(result, result2)
        # 断言结果与第二个预期均值 Series 对象相等
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize(
        "index",
        [
            # 创建包含三个不同类型索引的参数化测试
            [0, 1, 2, 3],
            ["a", "b", "c", "d"],
            [Timestamp(2021, 7, 28 + i) for i in range(4)],
        ],
    )
    # 定义测试方法，验证使用元组命名的 Series 对象的分组行为
    def test_groupby_series_named_with_tuple(self, frame_or_series, index):
        # 根据输入的 frame_or_series 和索引创建对象
        obj = frame_or_series([1, 2, 3, 4], index=index)
        # 创建包含相同索引的分组键 Series 对象
        groups = Series([1, 0, 1, 0], index=index, name=("a", "a"))
        # 对对象进行分组，并取出每组的最后一个元素
        result = obj.groupby(groups).last()
        # 创建预期的 Series 对象，其中包含预期的最后元素
        expected = frame_or_series([4, 3])
        # 设置预期结果的索引名称为元组 ("a", "a")
        expected.index.name = ("a", "a")
        # 断言结果与预期的最后元素 Series 对象相等
        tm.assert_equal(result, expected)

    # 定义测试方法，验证在使用函数作为分组依据时的行为
    def test_groupby_grouper_f_sanity_checked(self):
        # 创建包含日期范围的时间序列对象
        dates = date_range("01-Jan-2013", periods=12, freq="MS")
        # 创建具有标准正态分布随机数据的 Series 对象，使用日期作为索引
        ts = Series(np.random.default_rng(2).standard_normal(12), index=dates)

        # 对于传入的函数，简单检查其操作是否作用于整个索引
        msg = "'Timestamp' object is not subscriptable"
        # 断言传入的函数引发预期的 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            ts.groupby(lambda key: key[0:6])

        # 使用函数对 Series 进行分组，并计算每组的和
        result = ts.groupby(lambda x: x).sum()
        # 使用原始索引对 Series 进行分组，并计算每组的和
        expected = ts.groupby(ts.index).sum()
        # 移除预期结果的频率属性
        expected.index.freq = None
        # 断言结果与预期的总和 Series 对象相等
        tm.assert_series_equal(result, expected)

    # 定义测试方法，验证在使用 datetime 键进行分组时的行为
    def test_groupby_with_datetime_key(self):
        # 创建 DataFrame 对象，包含 id 列和日期时间列
        df = DataFrame(
            {
                "id": ["a", "b"] * 3,
                "b": date_range("2000-01-01", "2000-01-03", freq="9h"),
            }
        )
        # 创建 Grouper 对象，使用日期时间列 "b" 和频率 "D" 进行分组
        grouper = Grouper(key="b", freq="D")
        # 使用 Grouper 对象对 DataFrame 进行分组，同时使用 "id" 列
        gb = df.groupby([grouper, "id"])

        # 验证分组后的组数是否与预期的字典相等
        expected = {
            (Timestamp("2000-01-01"), "a"): [0, 2],
            (Timestamp("2000-01-01"), "b"): [1],
            (Timestamp("2000-01-02"), "a"): [4],
            (Timestamp("2000-01-02"), "b"): [3, 5],
        }
        # 断言分组后的组与预期的字典相等
        tm.assert_dict_equal(gb.groups, expected)

        # 验证分组后的组键数量是否与预期相等
        assert len(gb.groups.keys()) == 4

    # 定义测试方法，验证在多维输入上进行分组时的错误行为
    def test_grouping_error_on_multidim_input(self, df):
        # 创建预期的错误消息
        msg = "Grouper for '<class 'pandas.DataFrame'>' not 1-dimensional"
        # 断言在多维输入上创建 Grouping 对象时引发预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            Grouping(df.index, df[["A", "A"]])
    def test_multiindex_negative_level(self, multiindex_dataframe_random_data):
        # 测试用例：负索引级别分组
        # 对多级索引数据框根据最后一级索引进行分组求和
        result = multiindex_dataframe_random_data.groupby(level=-1).sum()
        # 期望结果：对多级索引数据框根据名为 "second" 的索引级别进行分组求和
        expected = multiindex_dataframe_random_data.groupby(level="second").sum()
        # 断言：验证结果与期望是否一致
        tm.assert_frame_equal(result, expected)

        # 对多级索引数据框根据倒数第二级索引进行分组求和
        result = multiindex_dataframe_random_data.groupby(level=-2).sum()
        # 期望结果：对多级索引数据框根据名为 "first" 的索引级别进行分组求和
        expected = multiindex_dataframe_random_data.groupby(level="first").sum()
        # 断言：验证结果与期望是否一致
        tm.assert_frame_equal(result, expected)

        # 对多级索引数据框根据倒数第二级和倒数第一级索引同时进行分组求和
        result = multiindex_dataframe_random_data.groupby(level=[-2, -1]).sum()
        # 期望结果：对索引进行排序后的多级索引数据框
        expected = multiindex_dataframe_random_data.sort_index()
        # 断言：验证结果与期望是否一致
        tm.assert_frame_equal(result, expected)

        # 对多级索引数据框根据倒数第一级和名为 "first" 的索引级别同时进行分组求和
        result = multiindex_dataframe_random_data.groupby(level=[-1, "first"]).sum()
        # 期望结果：对索引级别 "second" 和 "first" 进行分组求和
        expected = multiindex_dataframe_random_data.groupby(
            level=["second", "first"]
        ).sum()
        # 断言：验证结果与期望是否一致
        tm.assert_frame_equal(result, expected)

    def test_agg_with_dict_raises(self, df):
        # 测试用例：聚合操作使用字典引发异常
        df.columns = np.arange(len(df.columns))
        msg = "nested renamer is not supported"
        # 使用 pytest 断言期望引发 SpecificationError 异常，并且错误信息包含 "nested renamer is not supported"
        with pytest.raises(SpecificationError, match=msg):
            df.groupby(1, as_index=False)[2].agg({"Q": np.mean})

    def test_multiindex_columns_empty_level(self):
        # 测试用例：空索引级别处理
        lst = [["count", "values"], ["to filter", ""]]
        # 创建多级索引对象
        midx = MultiIndex.from_tuples(lst)

        # 创建包含单行数据的数据框
        df = DataFrame([[1, "A"]], columns=midx)

        # 根据 "to filter" 索引级别分组，并获取分组后的索引位置
        grouped = df.groupby("to filter").groups
        # 断言：验证分组结果中索引 "A" 的位置
        assert grouped["A"] == [0]

        # 根据空字符串索引级别分组，并获取分组后的索引位置
        grouped = df.groupby([("to filter", "")]).groups
        # 断言：验证分组结果中索引 "A" 的位置
        assert grouped["A"] == [0]

        # 创建包含多行数据的数据框
        df = DataFrame([[1, "A"], [2, "B"]], columns=midx)

        # 期望结果：根据 "to filter" 索引级别分组后的索引位置
        expected = df.groupby("to filter").groups
        # 实际结果：根据空字符串索引级别分组后的索引位置
        result = df.groupby([("to filter", "")]).groups
        # 断言：验证实际结果与期望结果是否一致
        assert result == expected

        # 创建包含多行相同数据的数据框
        df = DataFrame([[1, "A"], [2, "A"]], columns=midx)

        # 期望结果：根据 "to filter" 索引级别分组后的索引位置
        expected = df.groupby("to filter").groups
        # 实际结果：根据空字符串索引级别分组后的索引位置
        result = df.groupby([("to filter", "")]).groups
        # 使用 pytest 的断言方法，验证实际结果与期望结果是否一致
        tm.assert_dict_equal(result, expected)

    def test_groupby_multiindex_tuple(self):
        # 测试用例：使用元组进行多级索引分组
        # 创建包含多行数据的数据框，同时设置列的多级索引
        df = DataFrame(
            [[1, 2, 3, 4], [3, 4, 5, 6], [1, 4, 2, 3]],
            columns=MultiIndex.from_arrays([["a", "b", "b", "c"], [1, 1, 2, 2]]),
        )
        # 期望结果：根据 ("b", 1) 元组进行分组后的结果
        expected = df.groupby([("b", 1)]).groups
        # 实际结果：根据 ("b", 1) 元组进行分组后的结果
        result = df.groupby(("b", 1)).groups
        # 断言：验证实际结果与期望结果是否一致
        tm.assert_dict_equal(expected, result)

        # 创建包含相同数据的数据框，同时设置列的多级索引
        df2 = DataFrame(
            df.values,
            columns=MultiIndex.from_arrays(
                [["a", "b", "b", "c"], ["d", "d", "e", "e"]]
            ),
        )
        # 期望结果：根据 ("b", "d") 元组进行分组后的结果
        expected = df2.groupby([("b", "d")]).groups
        # 实际结果：根据 ("b", 1) 元组进行分组后的结果
        result = df.groupby(("b", 1)).groups
        # 断言：验证实际结果与期望结果是否一致
        tm.assert_dict_equal(expected, result)

        # 创建包含相同数据的数据框，同时设置列的多级索引
        df3 = DataFrame(df.values, columns=[("a", "d"), ("b", "d"), ("b", "e"), "c"])
        # 期望结果：根据 ("b", "d") 元组进行分组后的结果
        expected = df3.groupby([("b", "d")]).groups
        # 实际结果：根据 ("b", 1) 元组进行分组后的结果
        result = df.groupby(("b", 1)).groups
        # 断言：验证实际结果与期望结果是否一致
        tm.assert_dict_equal(expected, result)
    # 测试多级索引部分索引的 groupby 方法等价性
    def test_groupby_multiindex_partial_indexing_equivalence(self):
        # GH 17977
        # 创建包含多级索引的 DataFrame
        df = DataFrame(
            [[1, 2, 3, 4], [3, 4, 5, 6], [1, 4, 2, 3]],
            columns=MultiIndex.from_arrays([["a", "b", "b", "c"], [1, 1, 2, 2]]),
        )

        # 计算按照 [("a", 1)] 分组后 [("b", 1), ("b", 2)] 列的平均值，并进行断言比较
        expected_mean = df.groupby([("a", 1)])[[("b", 1), ("b", 2)]].mean()
        result_mean = df.groupby([("a", 1)])["b"].mean()
        tm.assert_frame_equal(expected_mean, result_mean)

        # 计算按照 [("a", 1)] 分组后 [("b", 1), ("b", 2)] 列的总和，并进行断言比较
        expected_sum = df.groupby([("a", 1)])[[("b", 1), ("b", 2)]].sum()
        result_sum = df.groupby([("a", 1)])["b"].sum()
        tm.assert_frame_equal(expected_sum, result_sum)

        # 计算按照 [("a", 1)] 分组后 [("b", 1), ("b", 2)] 列的计数，并进行断言比较
        expected_count = df.groupby([("a", 1)])[[("b", 1), ("b", 2)]].count()
        result_count = df.groupby([("a", 1)])["b"].count()
        tm.assert_frame_equal(expected_count, result_count)

        # 计算按照 [("a", 1)] 分组后 [("b", 1), ("b", 2)] 列的最小值，并进行断言比较
        expected_min = df.groupby([("a", 1)])[[("b", 1), ("b", 2)]].min()
        result_min = df.groupby([("a", 1)])["b"].min()
        tm.assert_frame_equal(expected_min, result_min)

        # 计算按照 [("a", 1)] 分组后 [("b", 1), ("b", 2)] 列的最大值，并进行断言比较
        expected_max = df.groupby([("a", 1)])[[("b", 1), ("b", 2)]].max()
        result_max = df.groupby([("a", 1)])["b"].max()
        tm.assert_frame_equal(expected_max, result_max)

        # 获取按照 [("a", 1)] 分组后 [("b", 1), ("b", 2)] 列的分组结果字典，并进行断言比较
        expected_groups = df.groupby([("a", 1)])[[("b", 1), ("b", 2)]].groups
        result_groups = df.groupby([("a", 1)])["b"].groups
        tm.assert_dict_equal(expected_groups, result_groups)

    # 测试 groupby 方法中的 level 参数
    def test_groupby_level(self, sort, multiindex_dataframe_random_data, df):
        # GH 17537
        # 获取包含多级索引的 DataFrame
        frame = multiindex_dataframe_random_data
        # 将 DataFrame 重置为普通索引的形式
        deleveled = frame.reset_index()

        # 按照第一级索引（level=0）对 frame 进行分组求和，并进行断言比较
        result0 = frame.groupby(level=0, sort=sort).sum()
        # 按照第二级索引（level=1）对 frame 进行分组求和，并进行断言比较
        result1 = frame.groupby(level=1, sort=sort).sum()

        # 根据重置后的 DataFrame 的 "first" 列值分组求和，并进行断言比较
        expected0 = frame.groupby(deleveled["first"].values, sort=sort).sum()
        # 根据重置后的 DataFrame 的 "second" 列值分组求和，并进行断言比较
        expected1 = frame.groupby(deleveled["second"].values, sort=sort).sum()

        # 设置预期的索引名称
        expected0.index.name = "first"
        expected1.index.name = "second"

        # 断言结果的索引名称与预期相同
        assert result0.index.name == "first"
        assert result1.index.name == "second"

        # 比较两个 DataFrame 是否相等，并进行断言比较
        tm.assert_frame_equal(result0, expected0)
        tm.assert_frame_equal(result1, expected1)
        
        # 断言结果的索引名称与 frame 的索引名称一致
        assert result0.index.name == frame.index.names[0]
        assert result1.index.name == frame.index.names[1]

        # 使用级别名称进行分组
        result0 = frame.groupby(level="first", sort=sort).sum()
        result1 = frame.groupby(level="second", sort=sort).sum()
        tm.assert_frame_equal(result0, expected0)
        tm.assert_frame_equal(result1, expected1)

        # 对非 MultiIndex 进行 level 参数不在 [-1, 0] 范围内的分组操作，预期引发 ValueError 异常
        msg = "level > 0 or level < -1 only valid with MultiIndex"
        with pytest.raises(ValueError, match=msg):
            df.groupby(level=1)
    # 定义测试函数，用于测试根据索引级别分组时的异常情况处理
    def test_groupby_level_index_names(self):
        # 创建一个DataFrame对象，包含"exp"列和"var1"列，将"exp"列设置为索引
        df = DataFrame({"exp": ["A"] * 3 + ["B"] * 3, "var1": range(6)}).set_index(
            "exp"
        )
        # 按索引级别"exp"进行分组操作
        df.groupby(level="exp")
        # 准备错误消息，用于断言抛出特定的 ValueError 异常
        msg = "level name foo is not the name of the index"
        # 使用 pytest.raises 断言捕获预期的 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            # 对DataFrame按不存在的索引级别"foo"进行分组，预期抛出 ValueError 异常
            df.groupby(level="foo")

    # 定义测试函数，用于测试带有缺失值情况下的索引级别分组
    def test_groupby_level_with_nas(self, sort):
        # 创建一个包含多级索引的MultiIndex对象
        index = MultiIndex(
            levels=[[1, 0], [0, 1, 2, 3]],
            codes=[[1, 1, 1, 1, 0, 0, 0, 0], [0, 1, 2, 3, 0, 1, 2, 3]],
        )

        # 创建一个Series对象，使用上述多级索引和一组数值
        s = Series(np.arange(8.0), index=index)
        # 对Series按第一级索引(level=0)进行分组，并求和
        result = s.groupby(level=0, sort=sort).sum()
        # 准备预期的结果Series对象
        expected = Series([6.0, 22.0], index=[0, 1])
        # 使用 tm.assert_series_equal 断言结果与预期相等
        tm.assert_series_equal(result, expected)

        # 创建另一个包含多级索引的MultiIndex对象，其中有一个无效的编码值(-1)
        index = MultiIndex(
            levels=[[1, 0], [0, 1, 2, 3]],
            codes=[[1, 1, 1, 1, -1, 0, 0, 0], [0, 1, 2, 3, 0, 1, 2, 3]],
        )

        # 创建另一个Series对象，使用上述多级索引和一组数值
        s = Series(np.arange(8.0), index=index)
        # 对Series按第一级索引(level=0)进行分组，并求和
        result = s.groupby(level=0, sort=sort).sum()
        # 准备预期的结果Series对象
        expected = Series([6.0, 18.0], index=[0.0, 1.0])
        # 使用 tm.assert_series_equal 断言结果与预期相等
        tm.assert_series_equal(result, expected)

    # 定义测试函数，用于测试分组操作时的参数设置
    def test_groupby_args(self, multiindex_dataframe_random_data):
        # 获取一个多级索引的DataFrame对象
        frame = multiindex_dataframe_random_data

        # 准备错误消息，用于断言抛出特定的 TypeError 异常
        msg = "You have to supply one of 'by' and 'level'"
        # 使用 pytest.raises 断言捕获预期的 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            # 调用groupby方法时没有传递任何参数，预期抛出 TypeError 异常
            frame.groupby()

        # 准备错误消息，用于断言抛出特定的 TypeError 异常
        msg = "You have to supply one of 'by' and 'level'"
        # 使用 pytest.raises 断言捕获预期的 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            # 调用groupby方法时传递了by=None和level=None的参数组合，预期抛出 TypeError 异常
            frame.groupby(by=None, level=None)

    # 使用pytest.mark.parametrize进行参数化测试
    @pytest.mark.parametrize(
        "sort,labels",
        [
            [True, [2, 2, 2, 0, 0, 1, 1, 3, 3, 3]],
            [False, [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]],
        ],
    )
    # 定义测试函数，用于测试分组操作时保持顺序
    def test_level_preserve_order(self, sort, labels, multiindex_dataframe_random_data):
        # 对多级索引的DataFrame对象按第一级索引(level=0)进行分组，根据参数化设置决定是否排序
        grouped = multiindex_dataframe_random_data.groupby(level=0, sort=sort)
        # 准备预期的分组标签数组
        exp_labels = np.array(labels, np.intp)
        # 使用 tm.assert_almost_equal 断言分组后的标识符与预期相等
        tm.assert_almost_equal(grouped._grouper.ids, exp_labels)

    # 定义测试函数，用于测试分组操作时的标签生成
    def test_grouping_labels(self, multiindex_dataframe_random_data):
        # 对多级索引的DataFrame对象按第一级索引(level=0)进行分组
        grouped = multiindex_dataframe_random_data.groupby(
            multiindex_dataframe_random_data.index.get_level_values(0)
        )
        # 准备预期的分组代码数组
        exp_labels = np.array([2, 2, 2, 0, 0, 1, 1, 3, 3, 3], dtype=np.intp)
        # 使用 tm.assert_almost_equal 断言分组后的代码与预期相等
        tm.assert_almost_equal(grouped._grouper.codes[0], exp_labels)
    def test_list_grouper_with_nat(self):
        # GH 14715
        # 创建一个包含日期范围的 DataFrame，从2011年1月1日开始，共365天，每天频率为一天
        df = DataFrame({"date": date_range("1/1/2011", periods=365, freq="D")})
        # 将最后一行的日期设置为 NaT (Not a Time)，即缺失时间数据
        df.iloc[-1] = pd.NaT
        # 创建一个 Grouper 对象，按照年初开始的年度频率进行分组
        grouper = Grouper(key="date", freq="YS")

        # 在一个列表分组中使用 Grouper
        result = df.groupby([grouper])
        # 预期结果是一个字典，键为 Timestamp("2011-01-01")，值为索引列表 [0, 1, ..., 363]
        expected = {Timestamp("2011-01-01"): Index(list(range(364)))}
        tm.assert_dict_equal(result.groups, expected)

        # 不使用列表进行分组的测试用例
        result = df.groupby(grouper)
        # 预期结果是一个字典，键为 Timestamp("2011-01-01")，值为 365
        expected = {Timestamp("2011-01-01"): 365}
        tm.assert_dict_equal(result.groups, expected)

    @pytest.mark.parametrize(
        "func,expected",
        [
            (
                "transform",
                Series(name=2, dtype=np.float64),
            ),
            (
                "agg",
                Series(
                    name=2, dtype=np.float64, index=Index([], dtype=np.float64, name=1)
                ),
            ),
            (
                "apply",
                Series(
                    name=2, dtype=np.float64, index=Index([], dtype=np.float64, name=1)
                ),
            ),
        ],
    )
    def test_evaluate_with_empty_groups(self, func, expected):
        # 26208
        # 测试处理空分组的 transform 操作
        # （不测试其他聚合函数，因为它们返回不同的索引对象）
        # 创建一个空的 DataFrame，包含两列 []
        df = DataFrame({1: [], 2: []})
        # 根据第一列分组，不保留分组键
        g = df.groupby(1, group_keys=False)
        # 对第二列使用传入的函数名 func 执行相应操作，lambda 函数直接返回输入值 x
        result = getattr(g[2], func)(lambda x: x)
        tm.assert_series_equal(result, expected)

    def test_groupby_empty(self):
        # https://github.com/pandas-dev/pandas/issues/27190
        # 创建一个空的 Series，数据类型为 float64
        s = Series([], name="name", dtype="float64")
        # 根据空列表分组，返回一个 GroupBy 对象
        gr = s.groupby([])

        # 计算分组的平均值
        result = gr.mean()
        # 期望结果是一个与 s 大小相同的空 Series，索引为一个空的整数索引
        expected = s.set_axis(Index([], dtype=np.intp))
        tm.assert_series_equal(result, expected)

        # 检查分组属性
        assert len(gr._grouper.groupings) == 1
        # 检查分组的 IDs，预期为空数组
        tm.assert_numpy_array_equal(
            gr._grouper.ids, np.array([], dtype=np.dtype(np.intp))
        )

        # 检查分组的数量
        assert gr._grouper.ngroups == 0

        # 检查名称
        gb = s.groupby(s)
        grouper = gb._grouper
        result = grouper.names
        expected = ["name"]
        assert result == expected

    def test_groupby_level_index_value_all_na(self):
        # issue 20519
        # 创建一个 DataFrame，包含三列 ["A", "B", "C"]，其中 "A" 和 "B" 列有 NaN 值
        df = DataFrame(
            [["x", np.nan, 10], [None, np.nan, 20]], columns=["A", "B", "C"]
        ).set_index(["A", "B"])
        # 根据 "A" 和 "B" 索引级别分组，并对 "C" 列求和
        result = df.groupby(level=["A", "B"]).sum()
        # 期望结果是一个空的 DataFrame，具有空的 MultiIndex
        expected = DataFrame(
            data=[],
            index=MultiIndex(
                levels=[Index(["x"], dtype="object"), Index([], dtype="float64")],
                codes=[[], []],
                names=["A", "B"],
            ),
            columns=["C"],
            dtype="int64",
        )
        tm.assert_frame_equal(result, expected)
    def test_groupby_multiindex_level_empty(self):
        # 创建一个包含三列的 DataFrame，数据用于测试多级索引的 groupby 操作
        df = DataFrame(
            [[123, "a", 1.0], [123, "b", 2.0]], columns=["id", "category", "value"]
        )
        # 将 id 和 category 列设置为多级索引
        df = df.set_index(["id", "category"])
        # 选择所有 value 列中小于 0 的行，此时为空的 DataFrame
        empty = df[df.value < 0]
        # 对空的 DataFrame 按 id 分组并求和
        result = empty.groupby("id").sum()
        # 创建一个预期结果 DataFrame，仅定义了其结构，无数据
        expected = DataFrame(
            dtype="float64",
            columns=["value"],
            index=Index([], dtype=np.int64, name="id"),
        )
        # 使用 pandas 的测试工具检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_groupby_tuple_keys_handle_multiindex(self):
        # 创建一个包含多列数据的 DataFrame，其中包括 category_tuple 列用于测试
        df = DataFrame(
            {
                "num1": [0, 8, 9, 4, 3, 3, 5, 9, 3, 6],
                "num2": [3, 8, 6, 4, 9, 2, 1, 7, 0, 9],
                "num3": [6, 5, 7, 8, 5, 1, 1, 10, 7, 8],
                "category_tuple": [
                    (0, 1),
                    (0, 1),
                    (0, 1),
                    (0, 4),
                    (2, 3),
                    (2, 3),
                    (2, 3),
                    (2, 3),
                    (5,),
                    (6,),
                ],
                "category_string": list("aaabbbbcde"),
            }
        )
        # 根据 category_tuple 列对 DataFrame 进行排序，并保存为预期的 DataFrame
        expected = df.sort_values(by=["category_tuple", "num1"])
        # 对 DataFrame 按 category_tuple 分组，并对每个组内的 num1 列排序
        result = df.groupby("category_tuple").apply(
            lambda x: x.sort_values(by="num1"), include_groups=False
        )
        # 将预期结果按 result 的列进行裁剪，以确保列顺序一致
        expected = expected[result.columns]
        # 使用 pandas 的测试工具检查 result 和 expected 是否相等，忽略索引顺序
        tm.assert_frame_equal(result.reset_index(drop=True), expected)
# 定义一个测试类 TestGetGroup，用于测试 DataFrame 的分组功能
class TestGetGroup:
    # 定义测试方法 test_get_group，测试基本的分组功能
    def test_get_group(self):
        # 创建一个 DataFrame 对象 df，包含 DATE、label 和 VAL 三列数据
        df = DataFrame(
            {
                "DATE": pd.to_datetime(
                    [
                        "10-Oct-2013",
                        "10-Oct-2013",
                        "10-Oct-2013",
                        "11-Oct-2013",
                        "11-Oct-2013",
                        "11-Oct-2013",
                    ]
                ),
                "label": ["foo", "foo", "bar", "foo", "foo", "bar"],
                "VAL": [1, 2, 3, 4, 5, 6],
            }
        )

        # 按照 DATE 列分组
        g = df.groupby("DATE")
        # 获取第一个分组的键值
        key = next(iter(g.groups))
        # 使用 get_group 方法获取指定分组的数据
        result1 = g.get_group(key)
        # 将 key 转换为 Timestamp 类型后再获取相同分组的数据
        result2 = g.get_group(Timestamp(key).to_pydatetime())
        # 将 key 转换为字符串后再获取相同分组的数据
        result3 = g.get_group(str(Timestamp(key)))
        # 断言三种方式获取的结果应该相等
        tm.assert_frame_equal(result1, result2)
        tm.assert_frame_equal(result1, result3)

        # 按照 DATE 和 label 两列分组
        g = df.groupby(["DATE", "label"])

        # 获取第一个分组的键值
        key = next(iter(g.groups))
        # 使用 get_group 方法获取指定分组的数据
        result1 = g.get_group(key)
        # 将 DATE 列的键值转换为 Timestamp 类型，再与 label 组成元组获取相同分组的数据
        result2 = g.get_group((Timestamp(key[0]).to_pydatetime(), key[1]))
        # 将 DATE 列的键值转换为字符串，再与 label 组成元组获取相同分组的数据
        result3 = g.get_group((str(Timestamp(key[0])), key[1]))
        # 断言三种方式获取的结果应该相等
        tm.assert_frame_equal(result1, result2)
        tm.assert_frame_equal(result1, result3)

        # 测试未能正确提供多个分组键时的异常情况
        msg = "must supply a same-length tuple with multiple keys"
        with pytest.raises(ValueError, match=msg):
            g.get_group("foo")
        with pytest.raises(ValueError, match=msg):
            g.get_group("foo")
        msg = "must supply a same-length tuple to get_group with multiple grouping keys"
        with pytest.raises(ValueError, match=msg):
            g.get_group(("foo", "bar", "baz"))

    # 定义测试方法 test_get_group_empty_bins，测试处理空分组情况
    def test_get_group_empty_bins(self, observed):
        # 创建一个包含单列数据的 DataFrame 对象 d
        d = DataFrame([3, 1, 7, 6])
        bins = [0, 5, 10, 15]
        # 根据数据的分布情况进行分组，observed 参数用于指定观察空分组的行为
        g = d.groupby(pd.cut(d[0], bins), observed=observed)

        # 获取指定区间的分组数据
        result = g.get_group(pd.Interval(0, 5))
        # 预期的结果 DataFrame
        expected = DataFrame([3, 1], index=[0, 1])
        # 断言获取的结果与预期的结果应该相等
        tm.assert_frame_equal(result, expected)

        # 测试未能找到指定区间的异常情况
        msg = r"Interval\(10, 15, closed='right'\)"
        with pytest.raises(KeyError, match=msg):
            g.get_group(pd.Interval(10, 15))
    def test_get_group_grouped_by_tuple(self):
        # GH 8121
        # 创建一个包含元组的DataFrame，并进行转置
        df = DataFrame([[(1,), (1, 2), (1,), (1, 2)]], index=["ids"]).T
        # 按照列 "ids" 进行分组
        gr = df.groupby("ids")
        # 创建预期的DataFrame，包含指定的元组
        expected = DataFrame({"ids": [(1,), (1,)]}, index=[0, 2])
        # 获取分组中元组为 (1,) 的结果
        result = gr.get_group((1,))
        # 使用测试工具比较结果和预期
        tm.assert_frame_equal(result, expected)

        # 创建一个包含日期时间的DataFrame
        dt = pd.to_datetime(["2010-01-01", "2010-01-02", "2010-01-01", "2010-01-02"])
        df = DataFrame({"ids": [(x,) for x in dt]})
        # 按照列 "ids" 进行分组
        gr = df.groupby("ids")
        # 获取分组中元组为 ("2010-01-01",) 的结果
        result = gr.get_group(("2010-01-01",))
        # 创建预期的DataFrame，包含指定的日期时间元组
        expected = DataFrame({"ids": [(dt[0],), (dt[0],)]}, index=[0, 2])
        # 使用测试工具比较结果和预期
        tm.assert_frame_equal(result, expected)

    def test_get_group_grouped_by_tuple_with_lambda(self):
        # GH 36158
        # 创建一个包含元组的DataFrame，其中元组由随机数生成
        df = DataFrame(
            {
                "Tuples": (
                    (x, y)
                    for x in [0, 1]
                    for y in np.random.default_rng(2).integers(3, 5, 5)
                )
            }
        )

        # 按照列 "Tuples" 进行分组
        gb = df.groupby("Tuples")
        # 使用 lambda 函数按照列 "Tuples" 进行分组
        gb_lambda = df.groupby(lambda x: df.iloc[x, 0])

        # 获取分组中第一个组的结果
        expected = gb.get_group(next(iter(gb.groups.keys())))
        # 获取使用 lambda 函数分组中第一个组的结果
        result = gb_lambda.get_group(next(iter(gb_lambda.groups.keys())))

        # 使用测试工具比较结果和预期
        tm.assert_frame_equal(result, expected)

    def test_groupby_with_empty(self):
        # 创建一个空的时间索引
        index = pd.DatetimeIndex(())
        # 创建空数据和对应的Series
        data = ()
        series = Series(data, index, dtype=object)
        # 使用频率 "D" 创建一个Grouper对象
        grouper = Grouper(freq="D")
        # 使用Grouper对象对Series进行分组
        grouped = series.groupby(grouper)
        # 断言分组后的第一个元素为None
        assert next(iter(grouped), None) is None

    def test_groupby_with_single_column(self):
        # 创建一个包含单列 "a" 的DataFrame
        df = DataFrame({"a": list("abssbab")})
        # 检查分组后列 "a" 中值为 "a" 的组的结果
        tm.assert_frame_equal(df.groupby("a").get_group("a"), df.iloc[[0, 5]])
        # GH 13530
        # 创建预期的空DataFrame
        exp = DataFrame(index=Index(["a", "b", "s"], name="a"), columns=[])
        # 使用 count 方法比较分组后的结果和预期
        tm.assert_frame_equal(df.groupby("a").count(), exp)
        # 使用 sum 方法比较分组后的结果和预期
        tm.assert_frame_equal(df.groupby("a").sum(), exp)

        # 创建预期的DataFrame，包含列 "a" 中的第二个元素
        exp = df.iloc[[3, 4, 5]]
        # 使用 nth 方法获取分组后的第一个元素
        tm.assert_frame_equal(df.groupby("a").nth(1), exp)

    def test_gb_key_len_equal_axis_len(self):
        # GH16843
        # 确保在分组时正确识别索引和列关键字，当关键字数等于分组轴长度时
        # 创建一个包含元组的DataFrame
        df = DataFrame(
            [["foo", "bar", "B", 1], ["foo", "bar", "B", 2], ["foo", "baz", "C", 3]],
            columns=["first", "second", "third", "one"],
        )
        # 设置索引为 "first" 和 "second"
        df = df.set_index(["first", "second"])
        # 按照列 "first", "second", "third" 进行分组，并计算组的大小
        df = df.groupby(["first", "second", "third"]).size()
        # 断言特定组的大小为预期值
        assert df.loc[("foo", "bar", "B")] == 2
        assert df.loc[("foo", "baz", "C")] == 1
# groups & iteration
# --------------------------------

class TestIteration:
    def test_groups(self, df):
        # 根据列"A"对DataFrame进行分组
        grouped = df.groupby(["A"])
        # 获取分组后的组信息
        groups = grouped.groups
        # 断言缓存有效性
        assert groups is grouped.groups  # caching works

        # 遍历每个组及其对应的索引
        for k, v in grouped.groups.items():
            # 断言每个分组中的"A"列的值都等于k
            assert (df.loc[v]["A"] == k).all()

        # 根据多列["A", "B"]对DataFrame进行分组
        grouped = df.groupby(["A", "B"])
        # 获取分组后的组信息
        groups = grouped.groups
        # 断言缓存有效性
        assert groups is grouped.groups  # caching works

        # 遍历每个组及其对应的索引
        for k, v in grouped.groups.items():
            # 断言每个分组中的"A"列的值都等于k[0]
            assert (df.loc[v]["A"] == k[0]).all()
            # 断言每个分组中的"B"列的值都等于k[1]
            assert (df.loc[v]["B"] == k[1]).all()

    def test_grouping_is_iterable(self, tsframe):
        # 这段代码路径没有在其他地方使用
        # 不确定其实用性
        # 根据lambda函数对时间序列DataFrame进行分组
        grouped = tsframe.groupby([lambda x: x.weekday(), lambda x: x.year])

        # 测试分组是否有效
        for g in grouped._grouper.groupings[0]:
            pass

    def test_multi_iter(self):
        # 创建一个Series对象
        s = Series(np.arange(6))
        # 创建两个分组键数组
        k1 = np.array(["a", "a", "a", "b", "b", "b"])
        k2 = np.array(["1", "2", "1", "2", "1", "2"])

        # 根据两个分组键对Series进行分组
        grouped = s.groupby([k1, k2])

        # 将分组结果转换为列表
        iterated = list(grouped)
        # 预期的分组结果
        expected = [
            ("a", "1", s[[0, 2]]),
            ("a", "2", s[[1]]),
            ("b", "1", s[[4]]),
            ("b", "2", s[[3, 5]]),
        ]
        # 遍历并断言每个分组的键和值是否与预期一致
        for i, ((one, two), three) in enumerate(iterated):
            e1, e2, e3 = expected[i]
            assert e1 == one
            assert e2 == two
            tm.assert_series_equal(three, e3)

    def test_multi_iter_frame(self, three_group):
        # 创建两个分组键数组
        k1 = np.array(["b", "b", "b", "a", "a", "a"])
        k2 = np.array(["1", "2", "1", "2", "1", "2"])
        # 创建一个DataFrame对象
        df = DataFrame(
            {
                "v1": np.random.default_rng(2).standard_normal(6),
                "v2": np.random.default_rng(2).standard_normal(6),
                "k1": k1,
                "k2": k2,
            },
            index=["one", "two", "three", "four", "five", "six"],
        )

        # 根据["k1", "k2"]对DataFrame进行分组
        grouped = df.groupby(["k1", "k2"])

        # 分组后的DataFrame会自动排序
        iterated = list(grouped)
        # 获取DataFrame的索引
        idx = df.index
        # 预期的分组结果
        expected = [
            ("a", "1", df.loc[idx[[4]]]),
            ("a", "2", df.loc[idx[[3, 5]]]),
            ("b", "1", df.loc[idx[[0, 2]]]),
            ("b", "2", df.loc[idx[[1]]]),
        ]
        # 遍历并断言每个分组的键和值是否与预期一致
        for i, ((one, two), three) in enumerate(iterated):
            e1, e2, e3 = expected[i]
            assert e1 == one
            assert e2 == two
            tm.assert_frame_equal(three, e3)

        # 不要遍历没有数据的分组
        df["k1"] = np.array(["b", "b", "b", "a", "a", "a"])
        df["k2"] = np.array(["1", "1", "1", "2", "2", "2"])
        grouped = df.groupby(["k1", "k2"])
        # 在DataFrameGroupBy上调用`dict`会引发TypeError，
        # 这里需要使用字典推导式来处理
        groups = {key: gp for key, gp in grouped}  # noqa: C416
        # 断言分组的数量是否符合预期
        assert len(groups) == 2
    def test_dictify(self, df):
        # 使用 groupby 方法将 DataFrame 按列"A"分组，然后转换为字典
        dict(iter(df.groupby("A")))
        # 使用 groupby 方法将 DataFrame 按列["A", "B"]分组，然后转换为字典
        dict(iter(df.groupby(["A", "B"])))
        # 使用 groupby 方法将 DataFrame 列"C"按列"A"分组，然后转换为字典
        dict(iter(df["C"].groupby(df["A"])))
        # 使用 groupby 方法将 DataFrame 列"C"按列["A", "B"]分组，然后转换为字典
        dict(iter(df["C"].groupby([df["A"], df["B"]])))
        # 使用 groupby 方法将 DataFrame 按列"A"分组，然后选取列"C"，再转换为字典
        dict(iter(df.groupby("A")["C"]))
        # 使用 groupby 方法将 DataFrame 按列["A", "B"]分组，然后选取列"C"，再转换为字典
        dict(iter(df.groupby(["A", "B"])["C"]))

    def test_groupby_with_small_elem(self):
        # GH 8542
        # 创建一个包含两行数据的 DataFrame
        df = DataFrame(
            {"event": ["start", "start"], "change": [1234, 5678]},
            index=pd.DatetimeIndex(["2014-09-10", "2013-10-10"]),
        )
        # 使用 groupby 方法按指定频率和列"event"分组
        grouped = df.groupby([Grouper(freq="ME"), "event"])
        # 断言分组的数量为2
        assert len(grouped.groups) == 2
        # 断言分组对象的唯一分组数量为2
        assert grouped.ngroups == 2
        # 断言指定日期和事件在分组中
        assert (Timestamp("2014-09-30"), "start") in grouped.groups
        assert (Timestamp("2013-10-31"), "start") in grouped.groups

        # 获取特定分组并断言与预期的 DataFrame 片段相等
        res = grouped.get_group((Timestamp("2014-09-30"), "start"))
        tm.assert_frame_equal(res, df.iloc[[0], :])
        res = grouped.get_group((Timestamp("2013-10-31"), "start"))
        tm.assert_frame_equal(res, df.iloc[[1], :])

        # 添加第三行数据的情况
        df = DataFrame(
            {"event": ["start", "start", "start"], "change": [1234, 5678, 9123]},
            index=pd.DatetimeIndex(["2014-09-10", "2013-10-10", "2014-09-15"]),
        )
        # 再次使用 groupby 方法按指定频率和列"event"分组
        grouped = df.groupby([Grouper(freq="ME"), "event"])
        # 断言分组的数量为2
        assert len(grouped.groups) == 2
        # 断言分组对象的唯一分组数量为2
        assert grouped.ngroups == 2
        # 断言指定日期和事件在分组中
        assert (Timestamp("2014-09-30"), "start") in grouped.groups
        assert (Timestamp("2013-10-31"), "start") in grouped.groups

        # 获取特定分组并断言与预期的 DataFrame 片段相等
        res = grouped.get_group((Timestamp("2014-09-30"), "start"))
        tm.assert_frame_equal(res, df.iloc[[0, 2], :])
        res = grouped.get_group((Timestamp("2013-10-31"), "start"))
        tm.assert_frame_equal(res, df.iloc[[1], :])

        # 再添加第四行数据的情况
        df = DataFrame(
            {"event": ["start", "start", "start"], "change": [1234, 5678, 9123]},
            index=pd.DatetimeIndex(["2014-09-10", "2013-10-10", "2014-08-05"]),
        )
        # 再次使用 groupby 方法按指定频率和列"event"分组
        grouped = df.groupby([Grouper(freq="ME"), "event"])
        # 断言分组的数量为3
        assert len(grouped.groups) == 3
        # 断言分组对象的唯一分组数量为3
        assert grouped.ngroups == 3
        # 断言指定日期和事件在分组中
        assert (Timestamp("2014-09-30"), "start") in grouped.groups
        assert (Timestamp("2013-10-31"), "start") in grouped.groups
        assert (Timestamp("2014-08-31"), "start") in grouped.groups

        # 获取特定分组并断言与预期的 DataFrame 片段相等
        res = grouped.get_group((Timestamp("2014-09-30"), "start"))
        tm.assert_frame_equal(res, df.iloc[[0], :])
        res = grouped.get_group((Timestamp("2013-10-31"), "start"))
        tm.assert_frame_equal(res, df.iloc[[1], :])
        res = grouped.get_group((Timestamp("2014-08-31"), "start"))
        tm.assert_frame_equal(res, df.iloc[[2], :])
    def test_grouping_string_repr(self):
        # 创建一个 MultiIndex 对象，其中包含两个数组：['A', 'A', 'B'] 和 ['a', 'b', 'a']
        mi = MultiIndex.from_arrays([list("AAB"), list("aba")])
        # 创建一个 DataFrame 对象，包含一个数据行 [1, 2, 3]，列使用 mi 作为列索引
        df = DataFrame([[1, 2, 3]], columns=mi)
        # 根据 df[("A", "a")] 列对 df 进行分组
        gr = df.groupby(df[("A", "a")])

        # 获取 gr 对象中 _grouper 属性的第一个 groupings 的字符串表示形式
        result = gr._grouper.groupings[0].__repr__()
        # 预期的字符串表示形式
        expected = "Grouping(('A', 'a'))"
        # 断言结果与预期相等
        assert result == expected
def test_grouping_by_key_is_in_axis():
    # GH#50413 - Groupers specified by key are in-axis
    # 创建一个 DataFrame 对象，包含三列，其中一列作为索引
    df = DataFrame({"a": [1, 1, 2], "b": [1, 1, 2], "c": [3, 4, 5]}).set_index("a")
    # 使用指定的键进行分组，确保分组不包含在轴内
    gb = df.groupby([Grouper(level="a"), Grouper(key="b")], as_index=False)
    # 断言第一个分组不在轴内
    assert not gb._grouper.groupings[0].in_axis
    # 断言第二个分组在轴内
    assert gb._grouper.groupings[1].in_axis

    # 对分组后的结果进行求和
    result = gb.sum()
    # 期望的结果 DataFrame
    expected = DataFrame({"a": [1, 2], "b": [1, 2], "c": [7, 5]})
    # 使用测试工具比较结果和期望值是否相等
    tm.assert_frame_equal(result, expected)
```