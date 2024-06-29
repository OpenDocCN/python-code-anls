# `D:\src\scipysrc\pandas\pandas\tests\reshape\test_melt.py`

```
import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架

import pandas as pd  # 导入Pandas库
from pandas import (  # 从Pandas库中导入特定函数和类
    DataFrame,
    Index,
    date_range,
    lreshape,
    melt,
    wide_to_long,
)
import pandas._testing as tm  # 导入Pandas测试模块


@pytest.fixture
def df():
    # 创建一个包含随机数据的DataFrame对象，具有特定的列和日期索引
    res = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    # 添加两列'id1'和'id2'到DataFrame，其值为对'A'和'B'列的条件判断结果转换为整数
    res["id1"] = (res["A"] > 0).astype(np.int64)
    res["id2"] = (res["B"] > 0).astype(np.int64)
    return res


@pytest.fixture
def df1():
    # 创建一个包含特定数据的DataFrame对象
    res = DataFrame(
        [
            [1.067683, -1.110463, 0.20867],
            [-1.321405, 0.368915, -1.055342],
            [-0.807333, 0.08298, -0.873361],
        ]
    )
    # 设置DataFrame的列名和列层级名称
    res.columns = [list("ABC"), list("abc")]
    res.columns.names = ["CAP", "low"]
    return res


@pytest.fixture
def var_name():
    # 返回一个字符串作为变量名
    return "var"


@pytest.fixture
def value_name():
    # 返回一个字符串作为值的名称
    return "val"


class TestMelt:
    def test_top_level_method(self, df):
        # 调用melt函数，并断言其结果DataFrame的列名
        result = melt(df)
        assert result.columns.tolist() == ["variable", "value"]

    def test_method_signatures(self, df, df1, var_name, value_name):
        # 测试不同参数组合下melt函数的返回结果是否相等
        tm.assert_frame_equal(df.melt(), melt(df))

        tm.assert_frame_equal(
            df.melt(id_vars=["id1", "id2"], value_vars=["A", "B"]),
            melt(df, id_vars=["id1", "id2"], value_vars=["A", "B"]),
        )

        tm.assert_frame_equal(
            df.melt(var_name=var_name, value_name=value_name),
            melt(df, var_name=var_name, value_name=value_name),
        )

        tm.assert_frame_equal(df1.melt(col_level=0), melt(df1, col_level=0))

    def test_default_col_names(self, df):
        # 测试不同id_vars参数下melt函数返回结果的列名是否符合预期
        result = df.melt()
        assert result.columns.tolist() == ["variable", "value"]

        result1 = df.melt(id_vars=["id1"])
        assert result1.columns.tolist() == ["id1", "variable", "value"]

        result2 = df.melt(id_vars=["id1", "id2"])
        assert result2.columns.tolist() == ["id1", "id2", "variable", "value"]

    def test_value_vars(self, df):
        # 测试指定id_vars和value_vars参数的情况下melt函数返回的DataFrame行数和内容是否符合预期
        result3 = df.melt(id_vars=["id1", "id2"], value_vars="A")
        assert len(result3) == 10

        result4 = df.melt(id_vars=["id1", "id2"], value_vars=["A", "B"])
        expected4 = DataFrame(
            {
                "id1": df["id1"].tolist() * 2,
                "id2": df["id2"].tolist() * 2,
                "variable": ["A"] * 10 + ["B"] * 10,
                "value": (df["A"].tolist() + df["B"].tolist()),
            },
            columns=["id1", "id2", "variable", "value"],
        )
        tm.assert_frame_equal(result4, expected4)

    @pytest.mark.parametrize("type_", (tuple, list, np.array))
    def test_value_vars_types(self, type_, df):
        # 测试函数，验证变量类型和值是否正确处理
        expected = DataFrame(
            {
                "id1": df["id1"].tolist() * 2,  # 创建包含两倍id1列内容的列表
                "id2": df["id2"].tolist() * 2,  # 创建包含两倍id2列内容的列表
                "variable": ["A"] * 10 + ["B"] * 10,  # 创建包含10个"A"和10个"B"的列表
                "value": (df["A"].tolist() + df["B"].tolist()),  # 将df中"A"和"B"列的值拼接成列表
            },
            columns=["id1", "id2", "variable", "value"],  # 指定DataFrame的列顺序
        )
        result = df.melt(id_vars=["id1", "id2"], value_vars=type_(("A", "B")))  # 使用melt函数将DataFrame进行重塑
        tm.assert_frame_equal(result, expected)  # 断言结果与期望值相等

    def test_vars_work_with_multiindex(self, df1):
        # 测试函数，验证处理具有多级索引的变量
        expected = DataFrame(
            {
                ("A", "a"): df1[("A", "a")],  # 选择df1中("A", "a")列的内容
                "CAP": ["B"] * len(df1),  # 创建包含len(df1)个"B"的列表
                "low": ["b"] * len(df1),  # 创建包含len(df1)个"b"的列表
                "value": df1[("B", "b")],  # 选择df1中("B", "b")列的内容
            },
            columns=[("A", "a"), "CAP", "low", "value"],  # 指定DataFrame的列顺序
        )

        result = df1.melt(id_vars=[("A", "a")], value_vars=[("B", "b")])  # 使用melt函数将DataFrame进行重塑
        tm.assert_frame_equal(result, expected)  # 断言结果与期望值相等

    @pytest.mark.parametrize(
        "id_vars, value_vars, col_level, expected",
        [
            (
                ["A"],
                ["B"],
                0,
                {
                    "A": {0: 1.067683, 1: -1.321405, 2: -0.807333},  # 创建包含索引0, 1, 2的"A"列值字典
                    "CAP": {0: "B", 1: "B", 2: "B"},  # 创建包含索引0, 1, 2的"CAP"列值字典
                    "value": {0: -1.110463, 1: 0.368915, 2: 0.08298},  # 创建包含索引0, 1, 2的"value"列值字典
                },
            ),
            (
                ["a"],
                ["b"],
                1,
                {
                    "a": {0: 1.067683, 1: -1.321405, 2: -0.807333},  # 创建包含索引0, 1, 2的"a"列值字典
                    "low": {0: "b", 1: "b", 2: "b"},  # 创建包含索引0, 1, 2的"low"列值字典
                    "value": {0: -1.110463, 1: 0.368915, 2: 0.08298},  # 创建包含索引0, 1, 2的"value"列值字典
                },
            ),
        ],
    )
    def test_single_vars_work_with_multiindex(
        self, id_vars, value_vars, col_level, expected, df1
    ):
        # 测试函数，验证处理具有多级索引的单一变量
        result = df1.melt(id_vars, value_vars, col_level=col_level)  # 使用melt函数将DataFrame进行重塑
        expected = DataFrame(expected)  # 创建期望结果的DataFrame
        tm.assert_frame_equal(result, expected)  # 断言结果与期望值相等

    @pytest.mark.parametrize(
        "id_vars, value_vars",
        [
            [("A", "a"), [("B", "b")]],  # 参数化测试：id_vars为("A", "a")，value_vars为[("B", "b")]
            [[("A", "a")], ("B", "b")],  # 参数化测试：id_vars为[("A", "a")]，value_vars为("B", "b")
            [("A", "a"), ("B", "b")],  # 参数化测试：id_vars为("A", "a")，value_vars为("B", "b")
        ],
    )
    def test_tuple_vars_fail_with_multiindex(self, id_vars, value_vars, df1):
        # 测试函数，验证处理具有多级索引且传递元组会失败的情况
        # 如果列有多级索引并且传递了元组作为id_vars或value_vars，则melt应该失败并显示信息性错误消息
        msg = r"(id|value)_vars must be a list of tuples when columns are a MultiIndex"
        with pytest.raises(ValueError, match=msg):
            df1.melt(id_vars=id_vars, value_vars=value_vars)  # 断言调用melt函数会引发ValueError异常，并匹配msg中的错误信息
    # 定义一个测试方法，用于验证数据框的变量名定制功能
    def test_custom_var_name(self, df, var_name):
        # 将数据框进行变形，使用给定的变量名作为新生成列的列名
        result5 = df.melt(var_name=var_name)
        # 断言变形后的列名列表是否符合预期
        assert result5.columns.tolist() == ["var", "value"]

        # 将数据框进行变形，保持 id1 列不变，其余列使用给定的变量名作为新生成列的列名
        result6 = df.melt(id_vars=["id1"], var_name=var_name)
        # 断言变形后的列名列表是否符合预期
        assert result6.columns.tolist() == ["id1", "var", "value"]

        # 将数据框进行变形，保持 id1 和 id2 列不变，其余列使用给定的变量名作为新生成列的列名
        result7 = df.melt(id_vars=["id1", "id2"], var_name=var_name)
        # 断言变形后的列名列表是否符合预期
        assert result7.columns.tolist() == ["id1", "id2", "var", "value"]

        # 将数据框进行变形，保持 id1 和 id2 列不变，仅处理列"A"，并使用给定的变量名作为新生成列的列名
        result8 = df.melt(id_vars=["id1", "id2"], value_vars="A", var_name=var_name)
        # 断言变形后的列名列表是否符合预期
        assert result8.columns.tolist() == ["id1", "id2", "var", "value"]

        # 将数据框进行变形，保持 id1 和 id2 列不变，处理列"A"和"B"，并使用给定的变量名作为新生成列的列名
        result9 = df.melt(
            id_vars=["id1", "id2"], value_vars=["A", "B"], var_name=var_name
        )
        # 构建预期结果数据框，包含重复的 id1 和 id2 列以及指定的变量名，以及变量值列
        expected9 = DataFrame(
            {
                "id1": df["id1"].tolist() * 2,
                "id2": df["id2"].tolist() * 2,
                var_name: ["A"] * 10 + ["B"] * 10,
                "value": (df["A"].tolist() + df["B"].tolist()),
            },
            columns=["id1", "id2", var_name, "value"],
        )
        # 使用测试框架中的方法比较实际结果和预期结果的一致性
        tm.assert_frame_equal(result9, expected9)

    # 定义一个测试方法，用于验证数据框的值名称定制功能
    def test_custom_value_name(self, df, value_name):
        # 将数据框进行变形，使用给定的值名称作为新生成值列的列名
        result10 = df.melt(value_name=value_name)
        # 断言变形后的列名列表是否符合预期
        assert result10.columns.tolist() == ["variable", "val"]

        # 将数据框进行变形，保持 id1 列不变，使用给定的值名称作为新生成值列的列名
        result11 = df.melt(id_vars=["id1"], value_name=value_name)
        # 断言变形后的列名列表是否符合预期
        assert result11.columns.tolist() == ["id1", "variable", "val"]

        # 将数据框进行变形，保持 id1 和 id2 列不变，使用给定的值名称作为新生成值列的列名
        result12 = df.melt(id_vars=["id1", "id2"], value_name=value_name)
        # 断言变形后的列名列表是否符合预期
        assert result12.columns.tolist() == ["id1", "id2", "variable", "val"]

        # 将数据框进行变形，保持 id1 和 id2 列不变，仅处理列"A"，并使用给定的值名称作为新生成值列的列名
        result13 = df.melt(
            id_vars=["id1", "id2"], value_vars="A", value_name=value_name
        )
        # 断言变形后的列名列表是否符合预期
        assert result13.columns.tolist() == ["id1", "id2", "variable", "val"]

        # 将数据框进行变形，保持 id1 和 id2 列不变，处理列"A"和"B"，并使用给定的值名称作为新生成值列的列名
        result14 = df.melt(
            id_vars=["id1", "id2"], value_vars=["A", "B"], value_name=value_name
        )
        # 构建预期结果数据框，包含重复的 id1 和 id2 列以及指定的变量名和值名称列
        expected14 = DataFrame(
            {
                "id1": df["id1"].tolist() * 2,
                "id2": df["id2"].tolist() * 2,
                "variable": ["A"] * 10 + ["B"] * 10,
                value_name: (df["A"].tolist() + df["B"].tolist()),
            },
            columns=["id1", "id2", "variable", value_name],
        )
        # 使用测试框架中的方法比较实际结果和预期结果的一致性
        tm.assert_frame_equal(result14, expected14)
    # 定义一个测试方法，用于测试数据框的变量和值的重塑操作
    def test_custom_var_and_value_name(self, df, value_name, var_name):
        # 对数据框进行变量(var)和值(value)的重塑操作，返回结果DataFrame
        result15 = df.melt(var_name=var_name, value_name=value_name)
        # 断言结果DataFrame的列名列表是否符合预期
        assert result15.columns.tolist() == ["var", "val"]

        # 对数据框进行带有指定id变量(id_vars)的变量(var)和值(value)的重塑操作，返回结果DataFrame
        result16 = df.melt(id_vars=["id1"], var_name=var_name, value_name=value_name)
        # 断言结果DataFrame的列名列表是否符合预期
        assert result16.columns.tolist() == ["id1", "var", "val"]

        # 对数据框进行带有多个指定id变量(id_vars)的变量(var)和值(value)的重塑操作，返回结果DataFrame
        result17 = df.melt(
            id_vars=["id1", "id2"], var_name=var_name, value_name=value_name
        )
        # 断言结果DataFrame的列名列表是否符合预期
        assert result17.columns.tolist() == ["id1", "id2", "var", "val"]

        # 对数据框进行带有指定value变量(value_vars)和id变量(id_vars)的变量(var)和值(value)的重塑操作，返回结果DataFrame
        result18 = df.melt(
            id_vars=["id1", "id2"],
            value_vars="A",
            var_name=var_name,
            value_name=value_name,
        )
        # 断言结果DataFrame的列名列表是否符合预期
        assert result18.columns.tolist() == ["id1", "id2", "var", "val"]

        # 对数据框进行带有指定value变量(value_vars)和id变量(id_vars)的变量(var)和值(value)的重塑操作，返回结果DataFrame
        result19 = df.melt(
            id_vars=["id1", "id2"],
            value_vars=["A", "B"],
            var_name=var_name,
            value_name=value_name,
        )
        # 生成预期的结果DataFrame
        expected19 = DataFrame(
            {
                "id1": df["id1"].tolist() * 2,
                "id2": df["id2"].tolist() * 2,
                var_name: ["A"] * 10 + ["B"] * 10,
                value_name: (df["A"].tolist() + df["B"].tolist()),
            },
            columns=["id1", "id2", var_name, value_name],
        )
        # 断言结果DataFrame是否与预期的DataFrame相等
        tm.assert_frame_equal(result19, expected19)

        # 复制数据框df，将列名设置为"foo"
        df20 = df.copy()
        df20.columns.name = "foo"
        # 对新数据框进行变量(var)和值(value)的重塑操作，返回结果DataFrame
        result20 = df20.melt()
        # 断言结果DataFrame的列名列表是否符合预期
        assert result20.columns.tolist() == ["foo", "value"]

    # 标记测试方法的参数化，测试列级别的变量和值的重塑操作
    @pytest.mark.parametrize("col_level", [0, "CAP"])
    def test_col_level(self, col_level, df1):
        # 对数据框进行列级别的变量(var)和值(value)的重塑操作，返回结果DataFrame
        res = df1.melt(col_level=col_level)
        # 断言结果DataFrame的列名列表是否符合预期
        assert res.columns.tolist() == ["CAP", "value"]

    # 测试多级索引数据框的变量和值的重塑操作
    def test_multiindex(self, df1):
        # 对多级索引数据框进行变量(var)和值(value)的重塑操作，返回结果DataFrame
        res = df1.melt()
        # 断言结果DataFrame的列名列表是否符合预期
        assert res.columns.tolist() == ["CAP", "low", "value"]

    # 标记测试方法的参数化，测试Pandas数据类型的变量和值的重塑操作
    @pytest.mark.parametrize(
        "col",
        [
            date_range("2010", periods=5, tz="US/Pacific"),
            pd.Categorical(["a", "b", "c", "a", "d"]),
            [0, 1, 0, 0, 0],
        ],
    )
    def test_pandas_dtypes(self, col):
        # GH 15785
        # 创建包含不同Pandas数据类型的系列(col)
        col = pd.Series(col)
        # 创建包含指定列的数据框(df)，并进行变量(var)和值(value)的重塑操作，返回结果DataFrame
        df = DataFrame(
            {"klass": range(5), "col": col, "attr1": [1, 0, 0, 0, 0], "attr2": col}
        )
        # 创建预期的值(expected_value)
        expected_value = pd.concat([pd.Series([1, 0, 0, 0, 0]), col], ignore_index=True)
        # 对数据框进行变量(var)和值(value)的重塑操作，返回结果DataFrame
        result = melt(
            df, id_vars=["klass", "col"], var_name="attribute", value_name="value"
        )
        # 创建预期的结果DataFrame(expected)
        expected = DataFrame(
            {
                0: list(range(5)) * 2,
                1: pd.concat([col] * 2, ignore_index=True),
                2: ["attr1"] * 5 + ["attr2"] * 5,
                3: expected_value,
            }
        )
        expected.columns = ["klass", "col", "attribute", "value"]
        # 断言结果DataFrame是否与预期的DataFrame相等
        tm.assert_frame_equal(result, expected)
    # 定义测试函数，用于测试数据列的融合操作是否正确保留类别信息
    def test_preserve_category(self):
        # GH 15853
        # 创建一个包含整数列和分类列的DataFrame对象
        data = DataFrame({"A": [1, 2], "B": pd.Categorical(["X", "Y"])})
        # 对数据进行融合操作，将'B'列作为id_vars，'A'列作为value_vars
        result = melt(data, ["B"], ["A"])
        # 期望的DataFrame结果，包含'B'列（分类列）、'variable'列和'value'列
        expected = DataFrame(
            {"B": pd.Categorical(["X", "Y"]), "variable": ["A", "A"], "value": [1, 2]}
        )

        # 断言函数结果与期望结果是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，验证在缺少列名的情况下是否会引发异常
    def test_melt_missing_columns_raises(self):
        # GH-23575
        # 这个测试用例确保 pandas 在尝试融合不存在的列名时会抛出错误

        # 生成一个随机数据的DataFrame
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)), columns=list("abcd")
        )

        # 尝试使用缺少的'value_vars'列名进行融合，预期会抛出KeyError异常
        msg = "The following id_vars or value_vars are not present in the DataFrame:"
        with pytest.raises(KeyError, match=msg):
            df.melt(["a", "b"], ["C", "d"])

        # 尝试使用缺少的'id_vars'列名进行融合，预期会抛出KeyError异常
        with pytest.raises(KeyError, match=msg):
            df.melt(["A", "b"], ["c", "d"])

        # 多列名缺失时，预期会抛出KeyError异常
        with pytest.raises(
            KeyError,
            match=msg,
        ):
            df.melt(["a", "b", "not_here", "or_there"], ["c", "d"])

        # 多级索引的融合，在多级融合时如果列名不存在，预期会抛出KeyError异常
        df.columns = [list("ABCD"), list("abcd")]
        with pytest.raises(KeyError, match=msg):
            df.melt([("E", "a")], [("B", "b")])
        # 单级融合时如果列名不存在，预期会抛出KeyError异常
        with pytest.raises(KeyError, match=msg):
            df.melt(["A"], ["F"], col_level=0)

    # 定义测试函数，验证在混合整数和字符串id_vars时的融合操作
    def test_melt_mixed_int_str_id_vars(self):
        # GH 29718
        # 创建一个包含整数和字符串作为列名的DataFrame对象
        df = DataFrame({0: ["foo"], "a": ["bar"], "b": [1], "d": [2]})
        # 对数据进行融合操作，id_vars包含整数和字符串，value_vars包含'b'和'd'
        result = melt(df, id_vars=[0, "a"], value_vars=["b", "d"])
        # 期望的DataFrame结果，包含0列、'a'列、'variable'列和'value'列
        expected = DataFrame(
            {0: ["foo"] * 2, "a": ["bar"] * 2, "variable": list("bd"), "value": [1, 2]}
        )
        # 断言函数结果与期望结果是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，验证在混合整数和字符串value_vars时的融合操作
    def test_melt_mixed_int_str_value_vars(self):
        # GH 29718
        # 创建一个包含整数和字符串作为列名的DataFrame对象
        df = DataFrame({0: ["foo"], "a": ["bar"]})
        # 对数据进行融合操作，value_vars包含整数和字符串列名
        result = melt(df, value_vars=[0, "a"])
        # 期望的DataFrame结果，包含'variable'列和'value'列
        expected = DataFrame({"variable": [0, "a"], "value": ["foo", "bar"]})
        # 断言函数结果与期望结果是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，验证是否正确忽略索引列进行融合操作
    def test_ignore_index(self):
        # GH 17440
        # 创建一个包含'foo'和'bar'的索引列的DataFrame对象
        df = DataFrame({"foo": [0], "bar": [1]}, index=["first"])
        # 对数据进行融合操作，不忽略索引列
        result = melt(df, ignore_index=False)
        # 期望的DataFrame结果，包含'variable'列和'value'列，与原索引列一致
        expected = DataFrame(
            {"variable": ["foo", "bar"], "value": [0, 1]}, index=["first", "first"]
        )
        # 断言函数结果与期望结果是否相等
        tm.assert_frame_equal(result, expected)
    # 测试忽略多重索引的情况，通过创建包含多重索引的DataFrame进行测试
    def test_ignore_multiindex(self):
        # GH 17440
        # 创建包含多重索引的DataFrame
        index = pd.MultiIndex.from_tuples(
            [("first", "second"), ("first", "third")], names=["baz", "foobar"]
        )
        df = DataFrame({"foo": [0, 1], "bar": [2, 3]}, index=index)
        # 调用melt函数，测试ignore_index=False的情况
        result = melt(df, ignore_index=False)

        # 预期的多重索引
        expected_index = pd.MultiIndex.from_tuples(
            [("first", "second"), ("first", "third")] * 2, names=["baz", "foobar"]
        )
        # 预期的DataFrame结果
        expected = DataFrame(
            {"variable": ["foo"] * 2 + ["bar"] * 2, "value": [0, 1, 2, 3]},
            index=expected_index,
        )

        # 断言结果DataFrame与预期DataFrame相等
        tm.assert_frame_equal(result, expected)

    # 测试忽略索引名称和类型的情况，通过创建包含特定索引的DataFrame进行测试
    def test_ignore_index_name_and_type(self):
        # GH 17440
        # 创建带有特定索引的DataFrame
        index = Index(["foo", "bar"], dtype="category", name="baz")
        df = DataFrame({"x": [0, 1], "y": [2, 3]}, index=index)
        # 调用melt函数，测试ignore_index=False的情况
        result = melt(df, ignore_index=False)

        # 预期的索引
        expected_index = Index(["foo", "bar"] * 2, dtype="category", name="baz")
        # 预期的DataFrame结果
        expected = DataFrame(
            {"variable": ["x", "x", "y", "y"], "value": [0, 1, 2, 3]},
            index=expected_index,
        )

        # 断言结果DataFrame与预期DataFrame相等
        tm.assert_frame_equal(result, expected)

    # 测试处理具有重复列名的情况，通过创建包含重复列名的DataFrame进行测试
    def test_melt_with_duplicate_columns(self):
        # GH#41951
        # 创建包含重复列名的DataFrame
        df = DataFrame([["id", 2, 3]], columns=["a", "b", "b"])
        # 调用DataFrame的melt方法，指定id_vars和value_vars，测试结果
        result = df.melt(id_vars=["a"], value_vars=["b"])

        # 预期的DataFrame结果
        expected = DataFrame(
            [["id", "b", 2], ["id", "b", 3]], columns=["a", "variable", "value"]
        )

        # 断言结果DataFrame与预期DataFrame相等
        tm.assert_frame_equal(result, expected)

    # 参数化测试不同dtype的情况，通过创建包含不同dtype的Series的DataFrame进行测试
    @pytest.mark.parametrize("dtype", ["Int8", "Int64"])
    def test_melt_ea_dtype(self, dtype):
        # GH#41570
        # 创建包含不同dtype的Series的DataFrame
        df = DataFrame(
            {
                "a": pd.Series([1, 2], dtype="Int8"),
                "b": pd.Series([3, 4], dtype=dtype),
            }
        )
        # 调用DataFrame的melt方法，测试结果
        result = df.melt()

        # 预期的DataFrame结果
        expected = DataFrame(
            {
                "variable": ["a", "a", "b", "b"],
                "value": pd.Series([1, 2, 3, 4], dtype=dtype),
            }
        )

        # 断言结果DataFrame与预期DataFrame相等
        tm.assert_frame_equal(result, expected)

    # 测试处理包含特定列的情况，通过创建包含特定列的DataFrame进行测试
    def test_melt_ea_columns(self):
        # GH 54297
        # 创建包含特定列的DataFrame
        df = DataFrame(
            {
                "A": {0: "a", 1: "b", 2: "c"},
                "B": {0: 1, 1: 3, 2: 5},
                "C": {0: 2, 1: 4, 2: 6},
            }
        )
        # 将列名称转换为指定类型
        df.columns = df.columns.astype("string[python]")
        # 调用DataFrame的melt方法，指定id_vars和value_vars，测试结果
        result = df.melt(id_vars=["A"], value_vars=["B"])

        # 预期的DataFrame结果
        expected = DataFrame(
            {
                "A": list("abc"),
                "variable": pd.Series(["B"] * 3, dtype="string[python]"),
                "value": [1, 3, 5],
            }
        )

        # 断言结果DataFrame与预期DataFrame相等
        tm.assert_frame_equal(result, expected)
    # 测试 DataFrame 的 melt 方法，验证是否保留日期时间信息
    def test_melt_preserves_datetime(self):
        # 创建包含日期时间数据的 DataFrame 对象
        df = DataFrame(
            data=[
                {
                    "type": "A0",
                    "start_date": pd.Timestamp("2023/03/01", tz="Asia/Tokyo"),
                    "end_date": pd.Timestamp("2023/03/10", tz="Asia/Tokyo"),
                },
                {
                    "type": "A1",
                    "start_date": pd.Timestamp("2023/03/01", tz="Asia/Tokyo"),
                    "end_date": pd.Timestamp("2023/03/11", tz="Asia/Tokyo"),
                },
            ],
            index=["aaaa", "bbbb"],
        )
        # 对 DataFrame 调用 melt 方法，将指定的列变量转换为行变量
        result = df.melt(
            id_vars=["type"],
            value_vars=["start_date", "end_date"],
            var_name="start/end",
            value_name="date",
        )
        # 创建预期结果的 DataFrame
        expected = DataFrame(
            {
                "type": {0: "A0", 1: "A1", 2: "A0", 3: "A1"},
                "start/end": {
                    0: "start_date",
                    1: "start_date",
                    2: "end_date",
                    3: "end_date",
                },
                "date": {
                    0: pd.Timestamp("2023-03-01 00:00:00+0900", tz="Asia/Tokyo"),
                    1: pd.Timestamp("2023-03-01 00:00:00+0900", tz="Asia/Tokyo"),
                    2: pd.Timestamp("2023-03-10 00:00:00+0900", tz="Asia/Tokyo"),
                    3: pd.Timestamp("2023-03-11 00:00:00+0900", tz="Asia/Tokyo"),
                },
            }
        )
        # 断言结果与预期结果是否相等
        tm.assert_frame_equal(result, expected)

    # 测试 DataFrame 的 melt 方法，验证是否允许非标量 id_vars
    def test_melt_allows_non_scalar_id_vars(self):
        # 创建包含非标量 id_vars 的 DataFrame 对象
        df = DataFrame(
            data={"a": [1, 2, 3], "b": [4, 5, 6]},
            index=["11", "22", "33"],
        )
        # 对 DataFrame 调用 melt 方法，将指定的列变量转换为行变量
        result = df.melt(
            id_vars="a",
            var_name=0,
            value_name=1,
        )
        # 创建预期结果的 DataFrame
        expected = DataFrame({"a": [1, 2, 3], 0: ["b"] * 3, 1: [4, 5, 6]})
        # 断言结果与预期结果是否相等
        tm.assert_frame_equal(result, expected)

    # 测试 DataFrame 的 melt 方法，验证是否允许非字符串的 var_name
    def test_melt_allows_non_string_var_name(self):
        # 创建包含非字符串 var_name 的 DataFrame 对象
        df = DataFrame(
            data={"a": [1, 2, 3], "b": [4, 5, 6]},
            index=["11", "22", "33"],
        )
        # 对 DataFrame 调用 melt 方法，将指定的列变量转换为行变量
        result = df.melt(
            id_vars=["a"],
            var_name=0,
            value_name=1,
        )
        # 创建预期结果的 DataFrame
        expected = DataFrame({"a": [1, 2, 3], 0: ["b"] * 3, 1: [4, 5, 6]})
        # 断言结果与预期结果是否相等
        tm.assert_frame_equal(result, expected)

    # 测试 DataFrame 的 melt 方法，验证非标量 var_name 是否会引发异常
    def test_melt_non_scalar_var_name_raises(self):
        # 创建包含非标量 var_name 的 DataFrame 对象
        df = DataFrame(
            data={"a": [1, 2, 3], "b": [4, 5, 6]},
            index=["11", "22", "33"],
        )
        # 使用 pytest 的上下文管理器确保抛出预期的 ValueError 异常
        with pytest.raises(ValueError, match=r".* must be a scalar."):
            # 对 DataFrame 调用 melt 方法，传入非标量 var_name 引发异常
            df.melt(id_vars=["a"], var_name=[1, 2])
    # 定义一个测试函数，用于测试多级索引列的变量名处理
    def test_melt_multiindex_columns_var_name(self):
        # GH 58033: 引用 GitHub issue 58033，描述这个测试的背景
        df = DataFrame({("A", "a"): [1], ("A", "b"): [2]})
        # 创建一个包含多级索引的 DataFrame 对象 df

        expected = DataFrame(
            [("A", "a", 1), ("A", "b", 2)], columns=["first", "second", "value"]
        )
        # 创建预期的 DataFrame 对象 expected，包含了列名 "first", "second", "value"

        # 使用 pandas 的 assert_frame_equal 方法比较 df 执行 melt 后的结果与 expected 是否相同
        tm.assert_frame_equal(df.melt(var_name=["first", "second"]), expected)

        # 使用 pandas 的 assert_frame_equal 方法比较 df 执行 melt 后的结果与 expected[["first", "value"]] 是否相同
        tm.assert_frame_equal(df.melt(var_name=["first"]), expected[["first", "value"]])

    # 定义另一个测试函数，用于测试多级索引列变量名过多的情况
    def test_melt_multiindex_columns_var_name_too_many(self):
        # GH 58033: 引用 GitHub issue 58033，描述这个测试的背景
        df = DataFrame({("A", "a"): [1], ("A", "b"): [2]})
        # 创建一个包含多级索引的 DataFrame 对象 df

        # 使用 pytest 的 pytest.raises 断言来捕获 ValueError 异常，并验证其匹配指定的错误消息
        with pytest.raises(
            ValueError, match="but the dataframe columns only have 2 levels"
        ):
            # 当 df 执行 melt 操作时，传入了三个变量名，但实际 DataFrame 的列级别只有 2 级，
            # 所以会引发 ValueError 异常，错误消息应包含 "but the dataframe columns only have 2 levels"
            df.melt(var_name=["first", "second", "third"])
class TestLreshape:
class TestWideToLong:
    def test_simple(self):
        # 使用随机数生成器创建指定种子的标准正态分布数组
        x = np.random.default_rng(2).standard_normal(3)
        # 创建包含多列数据的数据框
        df = DataFrame(
            {
                "A1970": {0: "a", 1: "b", 2: "c"},
                "A1980": {0: "d", 1: "e", 2: "f"},
                "B1970": {0: 2.5, 1: 1.2, 2: 0.7},
                "B1980": {0: 3.2, 1: 1.3, 2: 0.1},
                "X": dict(zip(range(3), x)),
            }
        )
        # 在数据框中添加一个 'id' 列，列值为索引值
        df["id"] = df.index
        # 期望的输出数据字典
        exp_data = {
            "X": x.tolist() + x.tolist(),
            "A": ["a", "b", "c", "d", "e", "f"],
            "B": [2.5, 1.2, 0.7, 3.2, 1.3, 0.1],
            "year": [1970, 1970, 1970, 1980, 1980, 1980],
            "id": [0, 1, 2, 0, 1, 2],
        }
        # 创建期望的数据框
        expected = DataFrame(exp_data)
        # 设置期望数据框的索引为 ('id', 'year')，并选择特定列 ['X', 'A', 'B']
        expected = expected.set_index(["id", "year"])[["X", "A", "B"]]
        # 调用 wide_to_long 函数进行测试
        result = wide_to_long(df, ["A", "B"], i="id", j="year")
        # 使用测试框架中的 assert 函数比较结果和期望
        tm.assert_frame_equal(result, expected)

    def test_stubs(self):
        # GH9204 wide_to_long 调用不应修改 'stubs' 列表
        # 创建包含两行数据的数据框，定义列名
        df = DataFrame([[0, 1, 2, 3, 8], [4, 5, 6, 7, 9]])
        df.columns = ["id", "inc1", "inc2", "edu1", "edu2"]
        # 定义 'stubs' 列表
        stubs = ["inc", "edu"]
        # 调用 wide_to_long 函数，使用指定的参数
        wide_to_long(df, stubs, i="id", j="age")
        # 使用 assert 语句检查 'stubs' 列表是否未被修改
        assert stubs == ["inc", "edu"]

    def test_separating_character(self):
        # GH14779
        # 使用随机数生成器创建指定种子的标准正态分布数组
        x = np.random.default_rng(2).standard_normal(3)
        # 创建包含多列数据的数据框，列名包含分隔符 '.'
        df = DataFrame(
            {
                "A.1970": {0: "a", 1: "b", 2: "c"},
                "A.1980": {0: "d", 1: "e", 2: "f"},
                "B.1970": {0: 2.5, 1: 1.2, 2: 0.7},
                "B.1980": {0: 3.2, 1: 1.3, 2: 0.1},
                "X": dict(zip(range(3), x)),
            }
        )
        # 在数据框中添加一个 'id' 列，列值为索引值
        df["id"] = df.index
        # 期望的输出数据字典
        exp_data = {
            "X": x.tolist() + x.tolist(),
            "A": ["a", "b", "c", "d", "e", "f"],
            "B": [2.5, 1.2, 0.7, 3.2, 1.3, 0.1],
            "year": [1970, 1970, 1970, 1980, 1980, 1980],
            "id": [0, 1, 2, 0, 1, 2],
        }
        # 创建期望的数据框
        expected = DataFrame(exp_data)
        # 设置期望数据框的索引为 ('id', 'year')，并选择特定列 ['X', 'A', 'B']
        expected = expected.set_index(["id", "year"])[["X", "A", "B"]]
        # 调用 wide_to_long 函数进行测试，指定分隔符参数 sep="."
        result = wide_to_long(df, ["A", "B"], i="id", j="year", sep=".")
        # 使用测试框架中的 assert 函数比较结果和期望
        tm.assert_frame_equal(result, expected)
    def test_escapable_characters(self):
        # 生成一个包含3个标准正态分布随机数的数组
        x = np.random.default_rng(2).standard_normal(3)
        # 创建一个DataFrame对象，包含列名带有括号和数字的数据
        df = DataFrame(
            {
                "A(quarterly)1970": {0: "a", 1: "b", 2: "c"},
                "A(quarterly)1980": {0: "d", 1: "e", 2: "f"},
                "B(quarterly)1970": {0: 2.5, 1: 1.2, 2: 0.7},
                "B(quarterly)1980": {0: 3.2, 1: 1.3, 2: 0.1},
                "X": dict(zip(range(3), x)),
            }
        )
        # 在DataFrame中添加一个名为'id'的新列，其值为索引值
        df["id"] = df.index
        # 期望的数据，包含多个列表，字典和数值数据
        exp_data = {
            "X": x.tolist() + x.tolist(),
            "A(quarterly)": ["a", "b", "c", "d", "e", "f"],
            "B(quarterly)": [2.5, 1.2, 0.7, 3.2, 1.3, 0.1],
            "year": [1970, 1970, 1970, 1980, 1980, 1980],
            "id": [0, 1, 2, 0, 1, 2],
        }
        # 创建一个期望的DataFrame对象
        expected = DataFrame(exp_data)
        # 将期望的DataFrame设置为以'id'和'year'为索引，选取特定列
        expected = expected.set_index(["id", "year"])[
            ["X", "A(quarterly)", "B(quarterly)"]
        ]
        # 调用wide_to_long函数，对df进行宽到长的转换，按'id'和'year'进行索引
        result = wide_to_long(df, ["A(quarterly)", "B(quarterly)"], i="id", j="year")
        # 使用tm.assert_frame_equal函数比较result和expected是否相等
        tm.assert_frame_equal(result, expected)

    def test_unbalanced(self):
        # 测试是否可以处理时间变量数量不一致的情况
        df = DataFrame(
            {
                "A2010": [1.0, 2.0],
                "A2011": [3.0, 4.0],
                "B2010": [5.0, 6.0],
                "X": ["X1", "X2"],
            }
        )
        # 在DataFrame中添加一个名为'id'的新列，其值为索引值
        df["id"] = df.index
        # 期望的数据，包含多个列表和数值数据
        exp_data = {
            "X": ["X1", "X2", "X1", "X2"],
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [5.0, 6.0, np.nan, np.nan],  # NaN表示缺失值
            "id": [0, 1, 0, 1],
            "year": [2010, 2010, 2011, 2011],
        }
        # 创建一个期望的DataFrame对象
        expected = DataFrame(exp_data)
        # 将期望的DataFrame设置为以'id'和'year'为索引，选取特定列
        expected = expected.set_index(["id", "year"])[["X", "A", "B"]]
        # 调用wide_to_long函数，对df进行宽到长的转换，按'id'和'year'进行索引
        result = wide_to_long(df, ["A", "B"], i="id", j="year")
        # 使用tm.assert_frame_equal函数比较result和expected是否相等
        tm.assert_frame_equal(result, expected)
    def test_character_overlap(self):
        # 测试处理在id_vars和value_vars中重叠字符的情况
        df = DataFrame(
            {
                "A11": ["a11", "a22", "a33"],
                "A12": ["a21", "a22", "a23"],
                "B11": ["b11", "b12", "b13"],
                "B12": ["b21", "b22", "b23"],
                "BB11": [1, 2, 3],
                "BB12": [4, 5, 6],
                "BBBX": [91, 92, 93],
                "BBBZ": [91, 92, 93],
            }
        )
        # 为DataFrame添加id列，id列的值为DataFrame的索引
        df["id"] = df.index
        expected = DataFrame(
            {
                "BBBX": [91, 92, 93, 91, 92, 93],
                "BBBZ": [91, 92, 93, 91, 92, 93],
                "A": ["a11", "a22", "a33", "a21", "a22", "a23"],
                "B": ["b11", "b12", "b13", "b21", "b22", "b23"],
                "BB": [1, 2, 3, 4, 5, 6],
                "id": [0, 1, 2, 0, 1, 2],
                "year": [11, 11, 11, 12, 12, 12],
            }
        )
        # 将期望的DataFrame按id和year列进行索引，并选择特定的列顺序
        expected = expected.set_index(["id", "year"])[["BBBX", "BBBZ", "A", "B", "BB"]]
        # 使用wide_to_long函数对df进行宽表转长表操作
        result = wide_to_long(df, ["A", "B", "BB"], i="id", j="year")
        # 使用assert_frame_equal断言函数比较结果DataFrame和期望DataFrame，按列名排序
        tm.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))

    def test_invalid_separator(self):
        # 如果提供了无效的分隔符，则返回一个空数据框
        sep = "nope!"
        df = DataFrame(
            {
                "A2010": [1.0, 2.0],
                "A2011": [3.0, 4.0],
                "B2010": [5.0, 6.0],
                "X": ["X1", "X2"],
            }
        )
        # 为DataFrame添加id列，id列的值为DataFrame的索引
        df["id"] = df.index
        exp_data = {
            "X": "",
            "A2010": [],
            "A2011": [],
            "B2010": [],
            "id": [],
            "year": [],
            "A": [],
            "B": [],
        }
        expected = DataFrame(exp_data).astype({"year": np.int64})
        # 将期望的DataFrame按id和year列进行索引，并选择特定的列顺序
        expected = expected.set_index(["id", "year"])[
            ["X", "A2010", "A2011", "B2010", "A", "B"]
        ]
        # 设置期望DataFrame的索引等级
        expected.index = expected.index.set_levels([0, 1], level=0)
        # 使用wide_to_long函数对df进行宽表转长表操作，指定分隔符为sep
        result = wide_to_long(df, ["A", "B"], i="id", j="year", sep=sep)
        # 使用assert_frame_equal断言函数比较结果DataFrame和期望DataFrame，按列名排序
        tm.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))
    def test_num_string_disambiguation(self):
        # 测试能够区分数字值变量和字符串值变量
        # 创建一个DataFrame对象，包含多列数据
        df = DataFrame(
            {
                "A11": ["a11", "a22", "a33"],
                "A12": ["a21", "a22", "a23"],
                "B11": ["b11", "b12", "b13"],
                "B12": ["b21", "b22", "b23"],
                "BB11": [1, 2, 3],
                "BB12": [4, 5, 6],
                "Arating": [91, 92, 93],
                "Arating_old": [91, 92, 93],
            }
        )
        # 添加一列'id'，其值为DataFrame的索引值
        df["id"] = df.index
        # 创建一个期望的DataFrame对象，包含多列数据
        expected = DataFrame(
            {
                "Arating": [91, 92, 93, 91, 92, 93],
                "Arating_old": [91, 92, 93, 91, 92, 93],
                "A": ["a11", "a22", "a33", "a21", "a22", "a23"],
                "B": ["b11", "b12", "b13", "b21", "b22", "b23"],
                "BB": [1, 2, 3, 4, 5, 6],
                "id": [0, 1, 2, 0, 1, 2],
                "year": [11, 11, 11, 12, 12, 12],
            }
        )
        # 设置期望DataFrame对象的索引为'id'和'year'列，选择部分列组成新的DataFrame
        expected = expected.set_index(["id", "year"])[
            ["Arating", "Arating_old", "A", "B", "BB"]
        ]
        # 调用函数wide_to_long处理DataFrame df，将宽格式转换为长格式，并将结果与期望的DataFrame比较
        result = wide_to_long(df, ["A", "B", "BB"], i="id", j="year")
        # 使用tm.assert_frame_equal函数比较结果DataFrame和期望DataFrame的排序后的内容
        tm.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))

    def test_invalid_suffixtype(self):
        # 如果所有的桩名称以字符串结尾，但假设存在数字后缀，则返回空数据框
        # 创建一个DataFrame对象，包含多列数据
        df = DataFrame(
            {
                "Aone": [1.0, 2.0],
                "Atwo": [3.0, 4.0],
                "Bone": [5.0, 6.0],
                "X": ["X1", "X2"],
            }
        )
        # 添加一列'id'，其值为DataFrame的索引值
        df["id"] = df.index
        # 创建一个期望的数据字典
        exp_data = {
            "X": "",
            "Aone": [],
            "Atwo": [],
            "Bone": [],
            "id": [],
            "year": [],
            "A": [],
            "B": [],
        }
        # 创建一个期望的DataFrame对象，并将其年份列转换为np.int64类型
        expected = DataFrame(exp_data).astype({"year": np.int64})

        # 设置期望DataFrame对象的索引为'id'和'year'列，将第一级索引设置为[0, 1]
        expected = expected.set_index(["id", "year"])
        # 调用函数wide_to_long处理DataFrame df，将宽格式转换为长格式，并将结果与期望的DataFrame比较
        result = wide_to_long(df, ["A", "B"], i="id", j="year")
        # 使用tm.assert_frame_equal函数比较结果DataFrame和期望DataFrame的排序后的内容
        tm.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))
    def test_multiple_id_columns(self):
        # 从http://www.ats.ucla.edu/stat/stata/modules/reshapel.htm获取的示例数据
        # 创建包含家庭ID、出生顺序、身高1和身高2的数据框
        df = DataFrame(
            {
                "famid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "birth": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "ht1": [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],
                "ht2": [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9],
            }
        )
        # 期望的结果数据框，包含合并后的身高、家庭ID、出生顺序和年龄
        expected = DataFrame(
            {
                "ht": [
                    2.8,
                    3.4,
                    2.9,
                    3.8,
                    2.2,
                    2.9,
                    2.0,
                    3.2,
                    1.8,
                    2.8,
                    1.9,
                    2.4,
                    2.2,
                    3.3,
                    2.3,
                    3.4,
                    2.1,
                    2.9,
                ],
                "famid": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
                "birth": [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3],
                "age": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            }
        )
        # 将期望的数据框设置为以家庭ID和出生顺序为索引，并只包含合并后的身高
        expected = expected.set_index(["famid", "birth", "age"])[["ht"]]
        # 调用函数wide_to_long，将df数据框从宽格式转换为长格式，根据家庭ID和出生顺序作为索引，年龄作为列名
        result = wide_to_long(df, "ht", i=["famid", "birth"], j="age")
        # 使用测试工具tm.assert_frame_equal检查结果是否与期望一致
        tm.assert_frame_equal(result, expected)

    def test_non_unique_idvars(self):
        # GH16382
        # 如果非唯一的id变量(i)被传入，抛出一个错误消息
        df = DataFrame(
            {"A_A1": [1, 2, 3, 4, 5], "B_B1": [1, 2, 3, 4, 5], "x": [1, 1, 1, 1, 1]}
        )
        msg = "the id variables need to uniquely identify each row"
        # 使用pytest.raises捕获期望的值错误，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            # 调用wide_to_long函数，将df数据框从宽格式转换为长格式，id变量为"x"，列名为"colname"
            wide_to_long(df, ["A_A", "B_B"], i="x", j="colname")
    # 定义测试方法，用于测试 wide_to_long 函数对数据框的转换
    def test_cast_j_int(self):
        # 创建一个包含演员和其 Facebook 点赞数的数据框
        df = DataFrame(
            {
                "actor_1": ["CCH Pounder", "Johnny Depp", "Christoph Waltz"],
                "actor_2": ["Joel David Moore", "Orlando Bloom", "Rory Kinnear"],
                "actor_fb_likes_1": [1000.0, 40000.0, 11000.0],
                "actor_fb_likes_2": [936.0, 5000.0, 393.0],
                "title": ["Avatar", "Pirates of the Caribbean", "Spectre"],
            }
        )

        # 创建预期结果数据框，包含演员、Facebook 点赞数、编号和电影标题，设定多重索引
        expected = DataFrame(
            {
                "actor": [
                    "CCH Pounder",
                    "Johnny Depp",
                    "Christoph Waltz",
                    "Joel David Moore",
                    "Orlando Bloom",
                    "Rory Kinnear",
                ],
                "actor_fb_likes": [1000.0, 40000.0, 11000.0, 936.0, 5000.0, 393.0],
                "num": [1, 1, 1, 2, 2, 2],
                "title": [
                    "Avatar",
                    "Pirates of the Caribbean",
                    "Spectre",
                    "Avatar",
                    "Pirates of the Caribbean",
                    "Spectre",
                ],
            }
        ).set_index(["title", "num"])

        # 使用 wide_to_long 函数将输入数据框 df 转换为长格式，指定列名前缀和分隔符，以索引列 "title" 和 "num"
        result = wide_to_long(
            df, ["actor", "actor_fb_likes"], i="title", j="num", sep="_"
        )

        # 断言转换后的结果与预期结果 expected 相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试 wide_to_long 函数处理标识符与列名相同的情况
    def test_identical_stubnames(self):
        # 创建一个包含带有年份后缀的数据框
        df = DataFrame(
            {
                "A2010": [1.0, 2.0],
                "A2011": [3.0, 4.0],
                "B2010": [5.0, 6.0],
                "A": ["X1", "X2"],
            }
        )
        # 设置错误消息，用于断言在列名与标识符重名时抛出 ValueError
        msg = "stubname can't be identical to a column name"
        # 使用 pytest 断言预期错误消息被抛出
        with pytest.raises(ValueError, match=msg):
            wide_to_long(df, ["A", "B"], i="A", j="colname")

    # 定义测试方法，用于测试 wide_to_long 函数处理非数值后缀的情况
    def test_nonnumeric_suffix(self):
        # 创建一个包含治疗方式和结果的数据框
        df = DataFrame(
            {
                "treatment_placebo": [1.0, 2.0],
                "treatment_test": [3.0, 4.0],
                "result_placebo": [5.0, 6.0],
                "A": ["X1", "X2"],
            }
        )
        # 创建预期结果数据框，包含索引 "A"、列名 "colname"、"result" 和 "treatment" 的长格式
        expected = DataFrame(
            {
                "A": ["X1", "X2", "X1", "X2"],
                "colname": ["placebo", "placebo", "test", "test"],
                "result": [5.0, 6.0, np.nan, np.nan],
                "treatment": [1.0, 2.0, 3.0, 4.0],
            }
        )
        # 将预期结果设定为多重索引，使用 wide_to_long 函数将 df 转换为长格式
        expected = expected.set_index(["A", "colname"])
        result = wide_to_long(
            df, ["result", "treatment"], i="A", j="colname", suffix="[a-z]+", sep="_"
        )
        # 断言转换后的结果与预期结果 expected 相等
        tm.assert_frame_equal(result, expected)
    # 定义测试函数，测试带有混合类型后缀的情况
    def test_mixed_type_suffix(self):
        # 创建一个 DataFrame 对象，包含多列数据
        df = DataFrame(
            {
                "A": ["X1", "X2"],  # 列'A'包含字符串 'X1' 和 'X2'
                "result_1": [0, 9],  # 列'result_1'包含整数值 0 和 9
                "result_foo": [5.0, 6.0],  # 列'result_foo'包含浮点数值 5.0 和 6.0
                "treatment_1": [1.0, 2.0],  # 列'treatment_1'包含浮点数值 1.0 和 2.0
                "treatment_foo": [3.0, 4.0],  # 列'treatment_foo'包含浮点数值 3.0 和 4.0
            }
        )
        # 创建预期的 DataFrame 对象，包含多列数据，并设置索引为 ('A', 'colname')
        expected = DataFrame(
            {
                "A": ["X1", "X2", "X1", "X2"],
                "colname": ["1", "1", "foo", "foo"],
                "result": [0.0, 9.0, 5.0, 6.0],  # 列'result'包含浮点数值 0.0, 9.0, 5.0, 6.0
                "treatment": [1.0, 2.0, 3.0, 4.0],  # 列'treatment'包含浮点数值 1.0, 2.0, 3.0, 4.0
            }
        ).set_index(["A", "colname"])  # 将 ('A', 'colname') 设为索引
        # 使用 wide_to_long 函数将宽格式的 df 转换为长格式，根据指定的列名后缀、分隔符和索引列名
        result = wide_to_long(
            df, ["result", "treatment"], i="A", j="colname", suffix=".+", sep="_"
        )
        # 使用 tm.assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，测试带有浮点数后缀的情况
    def test_float_suffix(self):
        # 创建一个 DataFrame 对象，包含多列数据
        df = DataFrame(
            {
                "treatment_1.1": [1.0, 2.0],  # 列'treatment_1.1'包含浮点数值 1.0 和 2.0
                "treatment_2.1": [3.0, 4.0],  # 列'treatment_2.1'包含浮点数值 3.0 和 4.0
                "result_1.2": [5.0, 6.0],  # 列'result_1.2'包含浮点数值 5.0 和 6.0
                "result_1": [0, 9],  # 列'result_1'包含整数值 0 和 9
                "A": ["X1", "X2"],  # 列'A'包含字符串 'X1' 和 'X2'
            }
        )
        # 创建预期的 DataFrame 对象，包含多列数据，并设置索引为 ('A', 'colname')
        expected = DataFrame(
            {
                "A": ["X1", "X2", "X1", "X2", "X1", "X2", "X1", "X2"],
                "colname": [1.2, 1.2, 1.0, 1.0, 1.1, 1.1, 2.1, 2.1],
                "result": [5.0, 6.0, 0.0, 9.0, np.nan, np.nan, np.nan, np.nan],  # 列'result'包含浮点数值和 NaN
                "treatment": [np.nan, np.nan, np.nan, np.nan, 1.0, 2.0, 3.0, 4.0],  # 列'treatment'包含浮点数值和 NaN
            }
        )
        # 将 expected 设置为以 ('A', 'colname') 为索引
        expected = expected.set_index(["A", "colname"])
        # 使用 wide_to_long 函数将宽格式的 df 转换为长格式，根据指定的列名后缀、分隔符和索引列名
        result = wide_to_long(
            df, ["result", "treatment"], i="A", j="colname", suffix="[0-9.]+", sep="_"
        )
        # 使用 tm.assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，测试列名是作为字符串传递的情况
    def test_col_substring_of_stubname(self):
        # GH22468
        # 当列名是作为字符串传递时，确保在 stubname 的子字符串匹配时不会引发 ValueError
        wide_data = {
            "node_id": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
            "A": {0: 0.80, 1: 0.0, 2: 0.25, 3: 1.0, 4: 0.81},
            "PA0": {0: 0.74, 1: 0.56, 2: 0.56, 3: 0.98, 4: 0.6},
            "PA1": {0: 0.77, 1: 0.64, 2: 0.52, 3: 0.98, 4: 0.67},
            "PA3": {0: 0.34, 1: 0.70, 2: 0.52, 3: 0.98, 4: 0.67},
        }
        # 创建 DataFrame 对象 wide_df，从 wide_data 字典中生成
        wide_df = DataFrame.from_dict(wide_data)
        # 创建预期的 DataFrame 对象，使用 wide_to_long 函数转换宽格式为长格式，根据指定的 stubnames 和索引列名
        expected = wide_to_long(wide_df, stubnames=["PA"], i=["node_id", "A"], j="time")
        # 使用 wide_to_long 函数将宽格式的 wide_df 转换为长格式，根据指定的 stubnames 和索引列名
        result = wide_to_long(wide_df, stubnames="PA", i=["node_id", "A"], j="time")
        # 使用 tm.assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，测试当结果值列名与数据框中已有的名称匹配时引发 ValueError 的情况
    def test_raise_of_column_name_value(self):
        # GH34731, 在版本 2.0 中生效
        # 当结果值列名与数据框中已有的名称匹配时，抛出 ValueError
        df = DataFrame({"col": list("ABC"), "value": range(10, 16, 2)})  # 创建 DataFrame 对象 df
        # 使用 pytest.raises 确保当 value_name="value" 时会抛出 ValueError，并匹配特定的错误消息
        with pytest.raises(
            ValueError, match=re.escape("value_name (value) cannot match")
        ):
            df.melt(id_vars="value", value_name="value")

    # 使用 pytest.mark.parametrize 装饰器为函数指定多个参数化的数据类型参数
    @pytest.mark.parametrize("dtype", ["O", "string"])
    # 定义一个测试函数，测试在缺少桩名时的行为，接受一个数据类型参数 dtype
    def test_missing_stubname(self, dtype):
        # GH46044: GitHub issue编号，用于跟踪相关问题
        # 创建一个 DataFrame，包含'id'、'a-1'和'a-2'三列数据
        df = DataFrame({"id": ["1", "2"], "a-1": [100, 200], "a-2": [300, 400]})
        # 将'id'列转换为指定的数据类型 dtype
        df = df.astype({"id": dtype})
        # 调用 wide_to_long 函数，将宽格式转换为长格式，基于'id'列作为主键，'num'作为新列的后缀，以'-'为分隔符
        result = wide_to_long(
            df,
            stubnames=["a", "b"],  # 桩名列表，指定需要转换的列名前缀
            i="id",  # 主键列名
            j="num",  # 新列名的后缀，从列名的末尾截取
            sep="-",  # 分隔符，用于分割后缀
        )
        # 创建一个 Index 对象，包含元组列表，每个元组由'id'和'num'组成
        index = Index(
            [("1", 1), ("2", 1), ("1", 2), ("2", 2)],
            name=("id", "num"),  # 设置索引的名称为'id'和'num'
        )
        # 创建一个期望的 DataFrame，包含两列'a'和'b'，每列有四行，'b'列用 NaN 填充
        expected = DataFrame(
            {"a": [100, 200, 300, 400], "b": [np.nan] * 4},
            index=index,  # 使用前面定义的索引对象作为索引
        )
        # 从期望的索引中获取第一级别（'id'）的值，并转换为指定的数据类型 dtype
        new_level = expected.index.levels[0].astype(dtype)
        # 设置期望 DataFrame 的索引的第一级别（'id'）为新的级别值
        expected.index = expected.index.set_levels(new_level, level=0)
        # 使用测试框架中的 assert_frame_equal 方法，比较结果 DataFrame 和期望 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
# 定义测试函数，用于测试 wide_to_long 函数的处理字符串列的功能
def test_wide_to_long_pyarrow_string_columns():
    # 要求导入 pyarrow 库，如果导入失败则跳过测试
    pytest.importorskip("pyarrow")
    
    # 创建一个 DataFrame 包含指定的列和数据
    df = DataFrame(
        {
            "ID": {0: 1},
            "R_test1": {0: 1},
            "R_test2": {0: 1},
            "R_test3": {0: 2},
            "D": {0: 1},
        }
    )
    
    # 将 DataFrame 的列名转换为指定类型 "string[pyarrow_numpy]"
    df.columns = df.columns.astype("string[pyarrow_numpy]")
    
    # 调用 wide_to_long 函数对 DataFrame 进行宽到长格式的转换
    result = wide_to_long(
        df, stubnames="R", i="ID", j="UNPIVOTED", sep="_", suffix=".*"
    )
    
    # 创建预期的 DataFrame，包含期望的数据结构
    expected = DataFrame(
        [[1, 1], [1, 1], [1, 2]],
        columns=Index(["D", "R"], dtype=object),
        index=pd.MultiIndex.from_arrays(
            [
                [1, 1, 1],
                Index(["test1", "test2", "test3"], dtype="string[pyarrow_numpy]"),
            ],
            names=["ID", "UNPIVOTED"],
        ),
    )
    
    # 使用 assert_frame_equal 函数比较实际结果和预期结果，确认测试是否通过
    tm.assert_frame_equal(result, expected)
```