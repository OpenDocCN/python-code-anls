# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_drop.py`

```
# 导入必要的库
import re  # 导入正则表达式模块
import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架

import pandas as pd  # 导入Pandas库
from pandas import (  # 从Pandas库中导入多个类和函数
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm  # 导入Pandas测试模块

# 使用pytest的@parametrize装饰器来参数化测试
@pytest.mark.parametrize(
    "msg,labels,level",
    [
        (r"labels \[4\] not found in level", 4, "a"),  # 参数化测试用例1
        (r"labels \[7\] not found in level", 7, "b"),  # 参数化测试用例2
    ],
)
def test_drop_raise_exception_if_labels_not_in_level(msg, labels, level):
    # 创建一个多级索引MultiIndex对象
    mi = MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]], names=["a", "b"])
    # 创建一个Series对象，使用上述多级索引作为索引
    s = Series([10, 20, 30], index=mi)
    # 创建一个DataFrame对象，使用上述多级索引作为索引
    df = DataFrame([10, 20, 30], index=mi)

    # 使用pytest的raises断言检查是否会抛出KeyError异常，并匹配给定的消息msg
    with pytest.raises(KeyError, match=msg):
        s.drop(labels, level=level)  # 测试Series对象的drop方法
    with pytest.raises(KeyError, match=msg):
        df.drop(labels, level=level)  # 测试DataFrame对象的drop方法


@pytest.mark.parametrize("labels,level", [(4, "a"), (7, "b")])
def test_drop_errors_ignore(labels, level):
    # 创建一个多级索引MultiIndex对象
    mi = MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]], names=["a", "b"])
    # 创建一个Series对象，使用上述多级索引作为索引
    s = Series([10, 20, 30], index=mi)
    # 创建一个DataFrame对象，使用上述多级索引作为索引
    df = DataFrame([10, 20, 30], index=mi)

    # 调用Series对象的drop方法，使用ignore错误处理选项，并将结果与期望的Series对象进行比较
    expected_s = s.drop(labels, level=level, errors="ignore")
    tm.assert_series_equal(s, expected_s)  # 使用Pandas测试模块断言Series对象相等

    # 调用DataFrame对象的drop方法，使用ignore错误处理选项，并将结果与期望的DataFrame对象进行比较
    expected_df = df.drop(labels, level=level, errors="ignore")
    tm.assert_frame_equal(df, expected_df)  # 使用Pandas测试模块断言DataFrame对象相等


def test_drop_with_non_unique_datetime_index_and_invalid_keys():
    # GH 30399

    # 定义一个具有唯一日期时间索引的DataFrame对象
    df = DataFrame(
        np.random.default_rng(2).standard_normal((5, 3)),
        columns=["a", "b", "c"],
        index=pd.date_range("2012", freq="h", periods=5),
    )
    # 创建一个具有非唯一日期时间索引的DataFrame对象的副本
    df = df.iloc[[0, 2, 2, 3]].copy()

    # 使用pytest的raises断言检查是否会抛出KeyError异常，并匹配给定的消息"not found in axis"
    with pytest.raises(KeyError, match="not found in axis"):
        df.drop(["a", "b"])  # 删除不在索引中存在的标签
    # 定义一个测试方法，用于测试数据框 DataFrame 的删除操作
    def test_drop_names(self):
        # 创建一个包含数据的 DataFrame，设定行索引和列索引的名称
        df = DataFrame(
            [[1, 2, 3], [3, 4, 5], [5, 6, 7]],
            index=["a", "b", "c"],
            columns=["d", "e", "f"],
        )
        # 设定行索引和列索引的名称为 "first" 和 "second"
        df.index.name, df.columns.name = "first", "second"
        
        # 按行删除索引为 "b" 的行，生成新的 DataFrame df_dropped_b
        df_dropped_b = df.drop("b")
        
        # 按列删除索引为 "e" 的列，生成新的 DataFrame df_dropped_e
        df_dropped_e = df.drop("e", axis=1)
        
        # 复制 DataFrame df 到 df_inplace_b 和 df_inplace_e
        df_inplace_b, df_inplace_e = df.copy(), df.copy()
        
        # 在 df_inplace_b 上就地删除索引为 "b" 的行，返回值应为 None
        return_value = df_inplace_b.drop("b", inplace=True)
        assert return_value is None
        
        # 在 df_inplace_e 上就地删除索引为 "e" 的列，返回值应为 None
        return_value = df_inplace_e.drop("e", axis=1, inplace=True)
        assert return_value is None
        
        # 验证四个对象（df_dropped_b、df_dropped_e、df_inplace_b、df_inplace_e）的行索引和列索引名称是否为 "first" 和 "second"
        for obj in (df_dropped_b, df_dropped_e, df_inplace_b, df_inplace_e):
            assert obj.index.name == "first"
            assert obj.columns.name == "second"
        
        # 验证 df 的列名是否为 ["d", "e", "f"]
        assert list(df.columns) == ["d", "e", "f"]

        # 设置预期的错误信息
        msg = r"\['g'\] not found in axis"
        
        # 使用 pytest 检查在删除不存在的索引 "g" 时是否引发 KeyError 错误
        with pytest.raises(KeyError, match=msg):
            df.drop(["g"])
        
        # 使用 pytest 检查在按列删除不存在的索引 "g" 时是否引发 KeyError 错误
        with pytest.raises(KeyError, match=msg):
            df.drop(["g"], axis=1)

        # 测试参数 errors="ignore" 的情况下，删除不存在的索引 "g"
        dropped = df.drop(["g"], errors="ignore")
        expected = Index(["a", "b", "c"], name="first")
        tm.assert_index_equal(dropped.index, expected)

        # 测试参数 errors="ignore" 的情况下，同时删除不存在的索引 "b" 和 "g"
        dropped = df.drop(["b", "g"], errors="ignore")
        expected = Index(["a", "c"], name="first")
        tm.assert_index_equal(dropped.index, expected)

        # 测试参数 axis=1 和 errors="ignore" 的情况下，删除不存在的列索引 "g"
        dropped = df.drop(["g"], axis=1, errors="ignore")
        expected = Index(["d", "e", "f"], name="second")
        tm.assert_index_equal(dropped.columns, expected)

        # 测试参数 axis=1 和 errors="ignore" 的情况下，同时删除不存在的列索引 "d" 和 "g"
        dropped = df.drop(["d", "g"], axis=1, errors="ignore")
        expected = Index(["e", "f"], name="second")
        tm.assert_index_equal(dropped.columns, expected)

        # GH 16398: 测试在不删除任何索引的情况下，是否能正确返回原始索引
        dropped = df.drop([], errors="ignore")
        expected = Index(["a", "b", "c"], name="first")
        tm.assert_index_equal(dropped.index, expected)
    # 定义一个测试函数，测试 DataFrame 对象的 drop 方法
    def test_drop(self):
        # 创建一个简单的 DataFrame 对象
        simple = DataFrame({"A": [1, 2, 3, 4], "B": [0, 1, 2, 3]})
        
        # 测试删除列 "A"，断言删除后的结果与预期的 DataFrame 相等
        tm.assert_frame_equal(simple.drop("A", axis=1), simple[["B"]])
        
        # 测试删除列 "A" 和 "B"，断言删除后的结果为空 DataFrame
        tm.assert_frame_equal(simple.drop(["A", "B"], axis="columns"), simple[[]])
        
        # 测试删除行索引为 [0, 1, 3] 的行，断言删除后的结果与预期的 DataFrame 相等
        tm.assert_frame_equal(simple.drop([0, 1, 3], axis=0), simple.loc[[2], :])
        
        # 测试删除行索引为 [0, 3] 的行，断言删除后的结果与预期的 DataFrame 相等
        tm.assert_frame_equal(simple.drop([0, 3], axis="index"), simple.loc[[1, 2], :])

        # 测试抛出 KeyError 的情况：试图删除不存在的列或行
        with pytest.raises(KeyError, match=r"\[5\] not found in axis"):
            simple.drop(5)
        with pytest.raises(KeyError, match=r"\['C'\] not found in axis"):
            simple.drop("C", axis=1)
        with pytest.raises(KeyError, match=r"\[5\] not found in axis"):
            simple.drop([1, 5])
        with pytest.raises(KeyError, match=r"\['C'\] not found in axis"):
            simple.drop(["A", "C"], axis=1)

        # GH 42881: 测试抛出 KeyError 的情况：试图删除不存在的多个列
        with pytest.raises(KeyError, match=r"\['C', 'D', 'F'\] not found in axis"):
            simple.drop(["C", "D", "F"], axis=1)

        # errors = 'ignore' 的情况下测试删除操作
        tm.assert_frame_equal(simple.drop(5, errors="ignore"), simple)
        tm.assert_frame_equal(
            simple.drop([0, 5], errors="ignore"), simple.loc[[1, 2, 3], :]
        )
        tm.assert_frame_equal(simple.drop("C", axis=1, errors="ignore"), simple)
        tm.assert_frame_equal(
            simple.drop(["A", "C"], axis=1, errors="ignore"), simple[["B"]]
        )

        # 测试删除非唯一列名的情况
        nu_df = DataFrame(
            list(zip(range(3), range(-3, 1), list("abc"))), columns=["a", "a", "b"]
        )
        tm.assert_frame_equal(nu_df.drop("a", axis=1), nu_df[["b"]])
        tm.assert_frame_equal(nu_df.drop("b", axis="columns"), nu_df["a"])
        tm.assert_frame_equal(nu_df.drop([]), nu_df)  # GH 16398

        # 设置索引和列名后的删除操作
        nu_df = nu_df.set_index(Index(["X", "Y", "X"]))
        nu_df.columns = list("abc")
        tm.assert_frame_equal(nu_df.drop("X", axis="rows"), nu_df.loc[["Y"], :])
        tm.assert_frame_equal(nu_df.drop(["X", "Y"], axis=0), nu_df.loc[[], :])

        # inplace=True 的情况下测试删除操作，并验证 inplace 修改是否正常
        # GH#5628
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=list("abc")
        )
        expected = df[~(df.b > 0)]
        return_value = df.drop(labels=df[df.b > 0].index, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(df, expected)
    def test_drop_multiindex_not_lexsorted(self, performance_warning):
        # GH#11640
        # 测试删除非词法排序多级索引的情况

        # 定义词法排序版本的多级索引
        lexsorted_mi = MultiIndex.from_tuples(
            [("a", ""), ("b1", "c1"), ("b2", "c2")], names=["b", "c"]
        )
        lexsorted_df = DataFrame([[1, 3, 4]], columns=lexsorted_mi)
        assert lexsorted_df.columns._is_lexsorted()

        # 定义非词法排序版本的数据框
        not_lexsorted_df = DataFrame(
            columns=["a", "b", "c", "d"], data=[[1, "b1", "c1", 3], [1, "b2", "c2", 4]]
        )
        # 对数据框进行透视操作
        not_lexsorted_df = not_lexsorted_df.pivot_table(
            index="a", columns=["b", "c"], values="d"
        )
        # 重置索引
        not_lexsorted_df = not_lexsorted_df.reset_index()
        assert not not_lexsorted_df.columns._is_lexsorted()

        # 期望的结果是从词法排序数据框中删除"a"列并转换为浮点数类型
        expected = lexsorted_df.drop("a", axis=1).astype(float)
        # 使用性能警告断言，测试非词法排序数据框删除"a"列的结果
        with tm.assert_produces_warning(performance_warning):
            result = not_lexsorted_df.drop("a", axis=1)

        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

    def test_drop_api_equivalence(self):
        # equivalence of the labels/axis and index/columns API's (GH#12392)
        # 测试标签/轴与索引/列API的等效性 (GH#12392)
        
        # 创建一个数据框
        df = DataFrame(
            [[1, 2, 3], [3, 4, 5], [5, 6, 7]],
            index=["a", "b", "c"],
            columns=["d", "e", "f"],
        )

        # 测试删除行"a"，使用标签API与索引API的等效性
        res1 = df.drop("a")
        res2 = df.drop(index="a")
        tm.assert_frame_equal(res1, res2)

        # 测试删除列"d"，使用轴API与列API的等效性
        res1 = df.drop("d", axis=1)
        res2 = df.drop(columns="d")
        tm.assert_frame_equal(res1, res2)

        # 测试删除列"e"，使用标签API与列API的等效性
        res1 = df.drop(labels="e", axis=1)
        res2 = df.drop(columns="e")
        tm.assert_frame_equal(res1, res2)

        # 测试删除行"a"，使用列表形式的索引API与索引API的等效性
        res1 = df.drop(["a"], axis=0)
        res2 = df.drop(index=["a"])
        tm.assert_frame_equal(res1, res2)

        # 测试先删除行"a"，再删除列"d"的操作，使用列表形式的索引API与列API的等效性
        res1 = df.drop(["a"], axis=0).drop(["d"], axis=1)
        res2 = df.drop(index=["a"], columns=["d"])
        tm.assert_frame_equal(res1, res2)

        # 测试当同时指定'labels'和'index'/'columns'时是否会引发 ValueError 异常
        msg = "Cannot specify both 'labels' and 'index'/'columns'"
        with pytest.raises(ValueError, match=msg):
            df.drop(labels="a", index="b")

        with pytest.raises(ValueError, match=msg):
            df.drop(labels="a", columns="b")

        # 测试当未指定'labels'、'index'或'columns'时是否会引发 ValueError 异常
        msg = "Need to specify at least one of 'labels', 'index' or 'columns'"
        with pytest.raises(ValueError, match=msg):
            df.drop(axis=1)
    # 测试函数：测试在删除重复索引时是否引发异常
    def test_raise_on_drop_duplicate_index(self, actual):
        # GH#19186
        # 如果实际数据的索引是 MultiIndex 类型，则设定 level 为 0，否则为 None
        level = 0 if isinstance(actual.index, MultiIndex) else None
        # 设置匹配模式，用于检查异常消息是否包含特定文本
        msg = re.escape("\"['c'] not found in axis\"")
        # 断言调用 actual.drop("c", level=level, axis=0) 时会抛出 KeyError 异常，并且异常消息匹配指定模式
        with pytest.raises(KeyError, match=msg):
            actual.drop("c", level=level, axis=0)
        # 断言调用 actual.T.drop("c", level=level, axis=1) 时会抛出 KeyError 异常，并且异常消息匹配指定模式
        with pytest.raises(KeyError, match=msg):
            actual.T.drop("c", level=level, axis=1)
        # 删除标签为 "c" 的行，忽略错误，返回预期结果
        expected_no_err = actual.drop("c", axis=0, level=level, errors="ignore")
        tm.assert_frame_equal(expected_no_err, actual)
        # 转置后删除标签为 "c" 的列，忽略错误，返回预期结果
        expected_no_err = actual.T.drop("c", axis=1, level=level, errors="ignore")
        tm.assert_frame_equal(expected_no_err.T, actual)

    @pytest.mark.parametrize("index", [[1, 2, 3], [1, 1, 2]])
    @pytest.mark.parametrize("drop_labels", [[], [1], [2]])
    # 测试函数：测试在 DataFrame 中删除空列表的标签
    def test_drop_empty_list(self, index, drop_labels):
        # GH#21494
        # 生成预期的索引，即排除掉 drop_labels 中存在的标签
        expected_index = [i for i in index if i not in drop_labels]
        # 创建 DataFrame 对象，并删除指定的标签
        frame = DataFrame(index=index).drop(drop_labels)
        # 断言删除后的 DataFrame 与预期的 DataFrame 相等
        tm.assert_frame_equal(frame, DataFrame(index=expected_index))

    @pytest.mark.parametrize("index", [[1, 2, 3], [1, 2, 2]])
    @pytest.mark.parametrize("drop_labels", [[1, 4], [4, 5]])
    # 测试函数：测试在 DataFrame 中删除非空列表的标签
    def test_drop_non_empty_list(self, index, drop_labels):
        # GH# 21494
        # 断言在尝试删除不存在于 DataFrame 中的标签时会引发 KeyError 异常
        with pytest.raises(KeyError, match="not found in axis"):
            DataFrame(index=index).drop(drop_labels)

    @pytest.mark.parametrize(
        "empty_listlike",
        [
            [],  # 空列表
            {},  # 空字典
            np.array([]),  # 空 NumPy 数组
            Series([], dtype="datetime64[ns]"),  # 空 Pandas Series
            Index([]),  # 空 Pandas Index
            DatetimeIndex([]),  # 空 Pandas DatetimeIndex
        ],
    )
    # 测试函数：测试在具有非唯一日期时间索引的 DataFrame 中删除空的列表样式对象
    def test_drop_empty_listlike_non_unique_datetime_index(self, empty_listlike):
        # GH#27994
        # 创建包含指定数据和索引的 DataFrame
        data = {"column_a": [5, 10], "column_b": ["one", "two"]}
        index = [Timestamp("2021-01-01"), Timestamp("2021-01-01")]
        df = DataFrame(data, index=index)

        # 生成预期结果，即应该返回原始的 DataFrame，因为传入的列表样式对象是空的
        expected = df.copy()
        # 删除指定的空列表样式对象，期望结果与原始 DataFrame 相等
        result = df.drop(empty_listlike)
        tm.assert_frame_equal(result, expected)
    def test_mixed_depth_drop(self):
        # 创建一个包含嵌套列表的变量arrays，每个子列表表示DataFrame的多层索引的一部分
        arrays = [
            ["a", "top", "top", "routine1", "routine1", "routine2"],
            ["", "OD", "OD", "result1", "result2", "result1"],
            ["", "wx", "wy", "", "", ""],
        ]

        # 将嵌套列表arrays转置并排序，然后使用转置后的元组列表创建MultiIndex索引对象index
        tuples = sorted(zip(*arrays))
        index = MultiIndex.from_tuples(tuples)

        # 创建一个4x6的DataFrame，其中数据由正态分布随机数生成，列使用前面创建的MultiIndex作为索引
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)

        # 删除列名为"a"的列，生成结果DataFrame result
        result = df.drop("a", axis=1)
        # 生成期望的DataFrame expected，删除列名为("a", "", "")的列
        expected = df.drop([("a", "", "")], axis=1)
        tm.assert_frame_equal(expected, result)

        # 删除列名为"top"的列，生成结果DataFrame result
        result = df.drop(["top"], axis=1)
        # 生成期望的DataFrame expected，删除列名为("top", "OD", "wx")和("top", "OD", "wy")的列
        expected = df.drop([("top", "OD", "wx")], axis=1)
        expected = expected.drop([("top", "OD", "wy")], axis=1)
        tm.assert_frame_equal(expected, result)

        # 删除列名为("top", "OD", "wx")的列，生成结果DataFrame result
        result = df.drop(("top", "OD", "wx"), axis=1)
        # 生成期望的DataFrame expected，删除列名为("top", "OD", "wx")的列
        expected = df.drop([("top", "OD", "wx")], axis=1)
        tm.assert_frame_equal(expected, result)

        # 生成期望的DataFrame expected，删除列名为("top", "OD", "wy")的列
        expected = df.drop([("top", "OD", "wy")], axis=1)
        # 生成期望的DataFrame expected，删除列名为"top"的列
        expected = df.drop("top", axis=1)

        # 删除多层索引第1级别中列名为"result1"的列，生成结果DataFrame result
        result = df.drop("result1", level=1, axis=1)
        # 生成期望的DataFrame expected，删除列名为("routine1", "result1", "")和("routine2", "result1", "")的列
        expected = df.drop(
            [("routine1", "result1", ""), ("routine2", "result1", "")], axis=1
        )
        tm.assert_frame_equal(expected, result)

    def test_drop_multiindex_other_level_nan(self):
        # 创建一个DataFrame df，包含列"A", "B", "C", "D"，并设置多层索引为["A", "B", "C"]，按索引排序
        df = (
            DataFrame(
                {
                    "A": ["one", "one", "two", "two"],
                    "B": [np.nan, 0.0, 1.0, 2.0],
                    "C": ["a", "b", "c", "c"],
                    "D": [1, 2, 3, 4],
                }
            )
            .set_index(["A", "B", "C"])
            .sort_index()
        )

        # 删除多层索引中第3级别中索引值为"c"的行，生成结果DataFrame result
        result = df.drop("c", level="C")
        # 生成期望的DataFrame expected，包含行索引为[("one", 0.0, "b"), ("one", np.nan, "a")]的数据
        expected = DataFrame(
            [2, 1],
            columns=["D"],
            index=MultiIndex.from_tuples(
                [("one", 0.0, "b"), ("one", np.nan, "a")], names=["A", "B", "C"]
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_drop_nonunique(self):
        # 创建一个DataFrame df，包含列"var1", "var2", "var3", "var4"
        df = DataFrame(
            [
                ["x-a", "x", "a", 1.5],
                ["x-a", "x", "a", 1.2],
                ["z-c", "z", "c", 3.1],
                ["x-a", "x", "a", 4.1],
                ["x-b", "x", "b", 5.1],
                ["x-b", "x", "b", 4.1],
                ["x-b", "x", "b", 2.2],
                ["y-a", "y", "a", 1.2],
                ["z-b", "z", "b", 2.1],
            ],
            columns=["var1", "var2", "var3", "var4"],
        )

        # 根据"var1"分组计算每组大小，选取大小为1的组的索引，生成drop_idx
        grp_size = df.groupby("var1").size()
        drop_idx = grp_size.loc[grp_size == 1]

        # 根据"var1", "var2", "var3"设置多层索引，生成idf
        idf = df.set_index(["var1", "var2", "var3"])

        # 删除idf中多层索引第0级别中在drop_idx.index中的行，重置索引，生成结果DataFrame result
        result = idf.drop(drop_idx.index, level=0).reset_index()
        # 生成期望的DataFrame expected，包含不在drop_idx.index中的行数据
        expected = df[-df.var1.isin(drop_idx.index)]

        result.index = expected.index

        tm.assert_frame_equal(result, expected)
    # 测试删除多重索引数据框中指定层级的行
    def test_drop_level(self, multiindex_dataframe_random_data):
        # 从测试数据中获取多重索引数据框
        frame = multiindex_dataframe_random_data

        # 删除第一层级中标签为 "bar" 和 "qux" 的行
        result = frame.drop(["bar", "qux"], level="first")
        # 期望结果是保留特定行的数据框
        expected = frame.iloc[[0, 1, 2, 5, 6]]
        tm.assert_frame_equal(result, expected)

        # 删除第二层级中标签为 "two" 的行
        result = frame.drop(["two"], level="second")
        # 期望结果是保留特定行的数据框
        expected = frame.iloc[[0, 2, 3, 6, 7, 9]]
        tm.assert_frame_equal(result, expected)

        # 对数据框转置后，删除第一层级中标签为 "bar" 和 "qux" 的列
        result = frame.T.drop(["bar", "qux"], axis=1, level="first")
        # 期望结果是保留特定列的数据框
        expected = frame.iloc[[0, 1, 2, 5, 6]].T
        tm.assert_frame_equal(result, expected)

        # 对数据框转置后，删除第二层级中标签为 "two" 的列
        result = frame.T.drop(["two"], axis=1, level="second")
        # 期望结果是保留特定列的数据框
        expected = frame.iloc[[0, 2, 3, 6, 7, 9]].T
        tm.assert_frame_equal(result, expected)

    # 测试在非唯一日期时间索引中删除指定层级的行
    def test_drop_level_nonunique_datetime(self):
        # 创建一个非唯一的日期时间索引
        idx = Index([2, 3, 4, 4, 5], name="id")
        idxdt = pd.to_datetime(
            [
                "2016-03-23 14:00",
                "2016-03-23 15:00",
                "2016-03-23 16:00",
                "2016-03-23 16:00",
                "2016-03-23 17:00",
            ]
        )
        # 创建一个数据框并设置非唯一日期时间索引
        df = DataFrame(np.arange(10).reshape(5, 2), columns=list("ab"), index=idx)
        df["tstamp"] = idxdt
        df = df.set_index("tstamp", append=True)
        ts = Timestamp("201603231600")
        assert df.index.is_unique is False

        # 删除指定日期时间层级的行
        result = df.drop(ts, level="tstamp")
        # 期望结果是保留特定行的数据框
        expected = df.loc[idx != 4]
        tm.assert_frame_equal(result, expected)

    # 测试在跨夏令时的时区感知时间戳中删除行
    def test_drop_tz_aware_timestamp_across_dst(self, frame_or_series):
        # 创建具有跨夏令时时区的日期时间索引
        start = Timestamp("2017-10-29", tz="Europe/Berlin")
        end = Timestamp("2017-10-29 04:00:00", tz="Europe/Berlin")
        index = pd.date_range(start, end, freq="15min")
        data = frame_or_series(data=[1] * len(index), index=index)
        # 删除指定时间戳
        result = data.drop(start)
        expected_start = Timestamp("2017-10-29 00:15:00", tz="Europe/Berlin")
        expected_idx = pd.date_range(expected_start, end, freq="15min")
        expected = frame_or_series(data=[1] * len(expected_idx), index=expected_idx)
        tm.assert_equal(result, expected)

    # 测试删除行后保留索引名称
    def test_drop_preserve_names(self):
        # 创建一个多重索引
        index = MultiIndex.from_arrays(
            [[0, 0, 0, 1, 1, 1], [1, 2, 3, 1, 2, 3]], names=["one", "two"]
        )
        # 创建一个数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((6, 3)), index=index)

        # 删除特定行
        result = df.drop([(0, 2)])
        # 断言结果数据框的索引名称保持不变
        assert result.index.names == ("one", "two")

    # 参数化测试，测试不同操作和是否原地修改的组合
    @pytest.mark.parametrize(
        "operation", ["__iadd__", "__isub__", "__imul__", "__ipow__"]
    )
    @pytest.mark.parametrize("inplace", [False, True])
    `
        def test_inplace_drop_and_operation(self, operation, inplace):
            # 测试 inplace 删除和其他操作的行为，GH#30484
            df = DataFrame({"x": range(5)})  # 创建一个包含一列数据的 DataFrame
            expected = df.copy()  # 复制 DataFrame，作为预期结果的基准
            df["y"] = range(5)  # 在 DataFrame 中添加新列 "y"
            y = df["y"]  # 获取新列 "y" 的引用
    
            with tm.assert_produces_warning(None):  # 确保操作不产生警告
                if inplace:
                    df.drop("y", axis=1, inplace=inplace)  # 在原地删除列 "y"
                else:
                    df = df.drop("y", axis=1, inplace=inplace)  # 删除列 "y"，并赋值给 df
    
                # 执行操作并检查结果
                getattr(y, operation)(1)  # 对列 y 执行指定的操作
                tm.assert_frame_equal(df, expected)  # 验证 DataFrame 是否与预期结果相同
    
        def test_drop_with_non_unique_multiindex(self):
            # 测试在非唯一 MultiIndex 上删除行，GH#36293
            mi = MultiIndex.from_arrays([["x", "y", "x"], ["i", "j", "i"]])  # 创建一个包含重复索引的 MultiIndex
            df = DataFrame([1, 2, 3], index=mi)  # 创建 DataFrame，索引为 MultiIndex
            result = df.drop(index="x")  # 删除索引为 "x" 的行
            expected = DataFrame([2], index=MultiIndex.from_arrays([["y"], ["j"]]))  # 定义预期结果 DataFrame
            tm.assert_frame_equal(result, expected)  # 验证结果是否与预期相同
    
        @pytest.mark.parametrize("indexer", [("a", "a"), [("a", "a")]])
        def test_drop_tuple_with_non_unique_multiindex(self, indexer):
            # 测试在非唯一 MultiIndex 上删除元组索引，GH#42771
            idx = MultiIndex.from_product([["a", "b"], ["a", "a"]])  # 创建一个 MultiIndex，包含笛卡尔积
            df = DataFrame({"x": range(len(idx))}, index=idx)  # 创建 DataFrame，索引为 idx
            result = df.drop(index=[("a", "a")])  # 删除索引为 ("a", "a") 的行
            expected = DataFrame(
                {"x": [2, 3]}, index=MultiIndex.from_tuples([("b", "a"), ("b", "a")])
            )  # 定义预期结果 DataFrame
            tm.assert_frame_equal(result, expected)  # 验证结果是否与预期相同
    
        def test_drop_with_duplicate_columns(self):
            df = DataFrame(
                [[1, 5, 7.0], [1, 5, 7.0], [1, 5, 7.0]], columns=["bar", "a", "a"]
            )  # 创建包含重复列名的 DataFrame
            result = df.drop(["a"], axis=1)  # 删除列名为 "a" 的列
            expected = DataFrame([[1], [1], [1]], columns=["bar"])  # 定义预期结果 DataFrame
            tm.assert_frame_equal(result, expected)  # 验证结果是否与预期相同
            result = df.drop("a", axis=1)  # 以字符串形式删除列名为 "a" 的列
            tm.assert_frame_equal(result, expected)  # 验证结果是否与预期相同
    
        def test_drop_with_duplicate_columns2(self):
            # 测试删除重复列名的情况，drop buggy GH#6240
            df = DataFrame(
                {
                    "A": np.random.default_rng(2).standard_normal(5),  # 创建包含随机数据的列 "A"
                    "B": np.random.default_rng(2).standard_normal(5),  # 创建包含随机数据的列 "B"
                    "C": np.random.default_rng(2).standard_normal(5),  # 创建包含随机数据的列 "C"
                    "D": ["a", "b", "c", "d", "e"],  # 创建包含字符串数据的列 "D"
                }
            )
    
            expected = df.take([0, 1, 1], axis=1)  # 预期结果 DataFrame，选择列 0、1 和 1
            df2 = df.take([2, 0, 1, 2, 1], axis=1)  # 创建 DataFrame df2，选择列 2、0、1、2 和 1
            result = df2.drop("C", axis=1)  # 删除列 "C"
            tm.assert_frame_equal(result, expected)  # 验证结果是否与预期相同
    
        def test_drop_inplace_no_leftover_column_reference(self):
            # 测试删除列后，确保没有剩余列引用，GH 13934
            df = DataFrame({"a": [1, 2, 3]}, columns=Index(["a"], dtype="object"))  # 创建 DataFrame
            a = df.a  # 获取列 "a" 的引用
            df.drop(["a"], axis=1, inplace=True)  # 在原地删除列 "a"
            tm.assert_index_equal(df.columns, Index([], dtype="object"))  # 验证 DataFrame 列为空
            a -= a.mean()  # 执行列 "a" 的均值减法
            tm.assert_index_equal(df.columns, Index([], dtype="object"))  # 再次验证 DataFrame 列为空
    
        def test_drop_level_missing_label_multiindex(self):
            # 测试在 MultiIndex 上删除不存在的标签，GH 18561
            df = DataFrame(index=MultiIndex.from_product([range(3), range(3)]))  # 创建一个 MultiIndex DataFrame
            with pytest.raises(KeyError, match="labels \\[5\\] not found in level"):  # 期望抛出 KeyError 异常
                df.drop(5, level=0)  # 尝试删除不存在的标签 5
    @pytest.mark.parametrize("idx, level", [(["a", "b"], 0), (["a"], None)])
    # 使用 pytest 的参数化装饰器，定义了两组参数 (["a", "b"], 0) 和 (["a"], None)，分别对应 idx 和 level
    def test_drop_index_ea_dtype(self, any_numeric_ea_dtype, idx, level):
        # GH#45860
        # 创建一个 DataFrame 对象 df，包含列 'a' 和 'b'，其中 'a' 列有值 [1, 2, 2, pd.NA]，'b' 列全为 100，数据类型为 any_numeric_ea_dtype
        df = DataFrame(
            {"a": [1, 2, 2, pd.NA], "b": 100}, dtype=any_numeric_ea_dtype
        ).set_index(idx)
        # 在 DataFrame df 上执行 drop 操作，删除索引中值为 2 和 pd.NA 的行，level 参数指定为给定的 level
        result = df.drop(Index([2, pd.NA]), level=level)
        # 创建一个预期的 DataFrame 对象 expected，包含列 'a' 和 'b'，'a' 列只有值 [1]，'b' 列为 100，数据类型为 any_numeric_ea_dtype，索引与 df 相同
        expected = DataFrame(
            {"a": [1], "b": 100}, dtype=any_numeric_ea_dtype
        ).set_index(idx)
        # 使用 assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_drop_parse_strings_datetime_index(self):
        # GH #5355
        # 创建一个 DataFrame 对象 df，包含列 'a' 和 'b'，索引使用 Timestamp 对象分别为 "2000-01-03" 和 "2000-01-04"
        df = DataFrame(
            {"a": [1, 2], "b": [1, 2]},
            index=[Timestamp("2000-01-03"), Timestamp("2000-01-04")],
        )
        # 在 DataFrame df 上执行 drop 操作，删除索引为 "2000-01-03" 的行，axis=0 表示按行操作
        result = df.drop("2000-01-03", axis=0)
        # 创建一个预期的 DataFrame 对象 expected，包含列 'a' 和 'b'，索引为 "2000-01-04"
        expected = DataFrame({"a": [2], "b": [2]}, index=[Timestamp("2000-01-04")])
        # 使用 assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
```