# `D:\src\scipysrc\pandas\pandas\tests\reshape\test_crosstab.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值数据
import pytest  # 导入 Pytest 测试框架

import pandas as pd  # 导入 Pandas 数据分析库
from pandas import (  # 导入 Pandas 中的特定对象
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    crosstab,
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具

@pytest.fixture  # 声明 Pytest 的 fixture，用于提供测试数据
def df():
    df = DataFrame(  # 创建一个 Pandas DataFrame 对象
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
            "D": np.random.default_rng(2).standard_normal(11),  # 生成随机数列 D
            "E": np.random.default_rng(2).standard_normal(11),  # 生成随机数列 E
            "F": np.random.default_rng(2).standard_normal(11),  # 生成随机数列 F
        }
    )

    return pd.concat([df, df], ignore_index=True)  # 返回经过拼接的 DataFrame 对象

class TestCrosstab:  # 定义测试类 TestCrosstab
    def test_crosstab_single(self, df):  # 定义测试方法 test_crosstab_single，接受 df fixture
        result = crosstab(df["A"], df["C"])  # 计算 A 列和 C 列的交叉表
        expected = df.groupby(["A", "C"]).size().unstack()  # 以 A 和 C 列分组，并计算组大小，将结果转换为二维表
        tm.assert_frame_equal(result, expected.fillna(0).astype(np.int64))  # 使用 Pandas 测试工具验证结果与预期是否相等

    def test_crosstab_multiple(self, df):  # 定义测试方法 test_crosstab_multiple，接受 df fixture
        result = crosstab(df["A"], [df["B"], df["C"]])  # 计算 A 列与组合列 B 和 C 的交叉表
        expected = df.groupby(["A", "B", "C"]).size()  # 以 A、B、C 列分组，并计算组大小
        expected = expected.unstack("B").unstack("C").fillna(0).astype(np.int64)  # 将结果重塑为三维表格，并填充 NaN 值，转换为整数类型
        tm.assert_frame_equal(result, expected)  # 使用 Pandas 测试工具验证结果与预期是否相等

        result = crosstab([df["B"], df["C"]], df["A"])  # 计算组合列 B 和 C 与 A 列的交叉表
        expected = df.groupby(["B", "C", "A"]).size()  # 以 B、C、A 列分组，并计算组大小
        expected = expected.unstack("A").fillna(0).astype(np.int64)  # 将结果重塑为二维表格，并填充 NaN 值，转换为整数类型
        tm.assert_frame_equal(result, expected)  # 使用 Pandas 测试工具验证结果与预期是否相等

    @pytest.mark.parametrize("box", [np.array, list, tuple])  # 使用 Pytest 的参数化装饰器，定义参数化测试
    # 测试函数：使用随机数生成器创建并填充指定大小的数组，并进行交叉制表分析
    def test_crosstab_ndarray(self, box):
        # GH 44076
        # 创建三个包含随机整数的数组，使用给定的随机数生成器填充
        a = box(np.random.default_rng(2).integers(0, 5, size=100))
        b = box(np.random.default_rng(2).integers(0, 3, size=100))
        c = box(np.random.default_rng(2).integers(0, 10, size=100))

        # 将数组a, b, c转换为DataFrame对象
        df = DataFrame({"a": a, "b": b, "c": c})

        # 进行交叉制表分析，计算结果并进行断言比较
        result = crosstab(a, [b, c], rownames=["a"], colnames=("b", "c"))
        expected = crosstab(df["a"], [df["b"], df["c"]])
        tm.assert_frame_equal(result, expected)

        # 进行第二种形式的交叉制表分析，计算结果并进行断言比较
        result = crosstab([b, c], a, colnames=["a"], rownames=("b", "c"))
        expected = crosstab([df["b"], df["c"]], df["a"])
        tm.assert_frame_equal(result, expected)

        # 对a和c进行交叉制表分析，指定索引和列名，并进行断言比较
        result = crosstab(a, c)
        expected = crosstab(df["a"], df["c"])
        # 设置预期结果的索引和列名
        expected.index.names = ["row_0"]
        expected.columns.names = ["col_0"]
        tm.assert_frame_equal(result, expected)

    # 测试函数：测试非对齐数据的交叉制表分析
    def test_crosstab_non_aligned(self):
        # GH 17005
        # 创建两个Series对象和一个NumPy数组
        a = Series([0, 1, 1], index=["a", "b", "c"])
        b = Series([3, 4, 3, 4, 3], index=["a", "b", "c", "d", "f"])
        c = np.array([3, 4, 3], dtype=np.int64)

        # 创建预期的DataFrame对象
        expected = DataFrame(
            [[1, 0], [1, 1]],
            index=Index([0, 1], name="row_0"),
            columns=Index([3, 4], name="col_0"),
        )

        # 进行交叉制表分析，计算结果并进行断言比较
        result = crosstab(a, b)
        tm.assert_frame_equal(result, expected)

        # 再次进行交叉制表分析，计算结果并进行断言比较
        result = crosstab(a, c)
        tm.assert_frame_equal(result, expected)

    # 测试函数：测试包含边际值的交叉制表分析
    def test_crosstab_margins(self):
        # 使用随机数生成器创建并填充指定大小的数组，并转换为DataFrame对象
        a = np.random.default_rng(2).integers(0, 7, size=100)
        b = np.random.default_rng(2).integers(0, 3, size=100)
        c = np.random.default_rng(2).integers(0, 5, size=100)

        df = DataFrame({"a": a, "b": b, "c": c})

        # 进行包含边际值的交叉制表分析，计算结果并进行断言比较
        result = crosstab(a, [b, c], rownames=["a"], colnames=("b", "c"), margins=True)

        # 断言结果的索引和列名
        assert result.index.names == ("a",)
        assert result.columns.names == ["b", "c"]

        # 计算行边际，并进行断言比较
        all_cols = result["All", ""]
        exp_cols = df.groupby(["a"]).size().astype("i8")
        exp_margin = Series([len(df)], index=Index(["All"], name="a"))
        exp_cols = pd.concat([exp_cols, exp_margin])
        exp_cols.name = ("All", "")
        tm.assert_series_equal(all_cols, exp_cols)

        # 计算列边际，并进行断言比较
        all_rows = result.loc["All"]
        exp_rows = df.groupby(["b", "c"]).size().astype("i8")
        exp_rows = pd.concat([exp_rows, Series([len(df)], index=[("All", "")])])
        exp_rows.name = "All"
        exp_rows = exp_rows.reindex(all_rows.index)
        exp_rows = exp_rows.fillna(0).astype(np.int64)
        tm.assert_series_equal(all_rows, exp_rows)
    def test_crosstab_margins_set_margin_name(self):
        # 创建随机整数数组 a, b, c，每个数组大小为 100
        a = np.random.default_rng(2).integers(0, 7, size=100)
        b = np.random.default_rng(2).integers(0, 3, size=100)
        c = np.random.default_rng(2).integers(0, 5, size=100)

        # 使用 a, b, c 创建 DataFrame 对象 df
        df = DataFrame({"a": a, "b": b, "c": c})

        # 调用 crosstab 函数生成交叉表 result
        result = crosstab(
            a,
            [b, c],
            rownames=["a"],
            colnames=("b", "c"),
            margins=True,
            margins_name="TOTAL",
        )

        # 断言结果 result 的索引名为 ("a")
        assert result.index.names == ("a",)
        # 断言结果 result 的列名为 ["b", "c"]
        assert result.columns.names == ["b", "c"]

        # 获取 result 中的 ("TOTAL", "") 列，并与预期的 exp_cols 比较
        all_cols = result["TOTAL", ""]
        exp_cols = df.groupby(["a"]).size().astype("i8")
        # 保持索引名
        exp_margin = Series([len(df)], index=Index(["TOTAL"], name="a"))
        exp_cols = pd.concat([exp_cols, exp_margin])
        exp_cols.name = ("TOTAL", "")

        # 使用 assert_series_equal 函数比较 all_cols 和 exp_cols
        tm.assert_series_equal(all_cols, exp_cols)

        # 获取 result 中的 "TOTAL" 行，并与预期的 exp_rows 比较
        all_rows = result.loc["TOTAL"]
        exp_rows = df.groupby(["b", "c"]).size().astype("i8")
        exp_rows = pd.concat([exp_rows, Series([len(df)], index=[("TOTAL", "")])])
        exp_rows.name = "TOTAL"

        # 重新索引 exp_rows 并填充 NaN 值，然后转换为 np.int64 类型
        exp_rows = exp_rows.reindex(all_rows.index)
        exp_rows = exp_rows.fillna(0).astype(np.int64)
        
        # 使用 assert_series_equal 函数比较 all_rows 和 exp_rows
        tm.assert_series_equal(all_rows, exp_rows)

        # 检查 margins_name 参数是否为字符串的异常情况
        msg = "margins_name argument must be a string"
        for margins_name in [666, None, ["a", "b"]]:
            with pytest.raises(ValueError, match=msg):
                crosstab(
                    a,
                    [b, c],
                    rownames=["a"],
                    colnames=("b", "c"),
                    margins=True,
                    margins_name=margins_name,
                )

    def test_crosstab_pass_values(self):
        # 创建随机整数数组 a, b, c 和随机标准正态分布数组 values，每个数组大小为 100
        a = np.random.default_rng(2).integers(0, 7, size=100)
        b = np.random.default_rng(2).integers(0, 3, size=100)
        c = np.random.default_rng(2).integers(0, 5, size=100)
        values = np.random.default_rng(2).standard_normal(100)

        # 调用 crosstab 函数生成交叉表 table
        table = crosstab(
            [a, b], c, values, aggfunc="sum", rownames=["foo", "bar"], colnames=["baz"]
        )

        # 使用 a, b, c, values 创建 DataFrame 对象 df
        df = DataFrame({"foo": a, "bar": b, "baz": c, "values": values})

        # 使用 pivot_table 函数生成预期的交叉表 expected，并与 table 进行比较
        expected = df.pivot_table(
            "values", index=["foo", "bar"], columns="baz", aggfunc="sum"
        )
        # 使用 assert_frame_equal 函数比较 table 和 expected
        tm.assert_frame_equal(table, expected)
    def test_crosstab_dropna(self):
        # GH 3820
        # 创建包含字符串数组的 NumPy 数组 a，表示类别数据
        a = np.array(["foo", "foo", "foo", "bar", "bar", "foo", "foo"], dtype=object)
        # 创建包含字符串数组的 NumPy 数组 b，表示类别数据
        b = np.array(["one", "one", "two", "one", "two", "two", "two"], dtype=object)
        # 创建包含字符串数组的 NumPy 数组 c，表示类别数据
        c = np.array(
            ["dull", "dull", "dull", "dull", "dull", "shiny", "shiny"], dtype=object
        )
        # 使用 crosstab 函数计算交叉表，dropna 参数设置为 False，保留 NaN 值
        res = crosstab(a, [b, c], rownames=["a"], colnames=["b", "c"], dropna=False)
        # 创建 MultiIndex 对象 m，表示多级索引
        m = MultiIndex.from_tuples(
            [("one", "dull"), ("one", "shiny"), ("two", "dull"), ("two", "shiny")],
            names=["b", "c"],
        )
        # 断言结果 res 的列索引与预期的 m 相等
        tm.assert_index_equal(res.columns, m)

    def test_crosstab_no_overlap(self):
        # GS 10291
        # 创建包含整数数据的 Pandas Series 对象 s1
        s1 = Series([1, 2, 3], index=[1, 2, 3])
        # 创建包含整数数据的 Pandas Series 对象 s2
        s2 = Series([4, 5, 6], index=[4, 5, 6])
        # 使用 crosstab 函数计算 s1 和 s2 的交叉表
        actual = crosstab(s1, s2)
        # 创建一个空的 DataFrame 对象 expected，表示预期的结果
        expected = DataFrame(
            index=Index([], dtype="int64", name="row_0"),
            columns=Index([], dtype="int64", name="col_0"),
        )
        # 断言计算结果 actual 与预期结果 expected 相等
        tm.assert_frame_equal(actual, expected)

    def test_margin_dropna(self):
        # GH 12577
        # 创建包含字典数据的 Pandas DataFrame 对象 df
        # DataFrame 包含两列 'a' 和 'b'，其中 'a' 列包含数值和 NaN，'b' 列包含整数数据
        df = DataFrame({"a": [1, 2, 2, 2, 2, np.nan], "b": [3, 3, 4, 4, 4, 4]})
        # 使用 crosstab 函数计算 df 的交叉表，设置 margins=True 和 dropna=True
        actual = crosstab(df.a, df.b, margins=True, dropna=True)
        # 创建一个包含预期结果的 DataFrame 对象 expected
        expected = DataFrame([[1, 0, 1], [1, 3, 4], [2, 3, 5]])
        expected.index = Index([1.0, 2.0, "All"], name="a")
        expected.columns = Index([3, 4, "All"], name="b")
        # 断言计算结果 actual 与预期结果 expected 相等
        tm.assert_frame_equal(actual, expected)

    def test_margin_dropna2(self):
        # 创建包含字典数据的 Pandas DataFrame 对象 df
        # DataFrame 包含两列 'a' 和 'b'，其中 'a' 列包含数值和 NaN，'b' 列包含数值和 NaN
        df = DataFrame(
            {"a": [1, np.nan, np.nan, np.nan, 2, np.nan], "b": [3, np.nan, 4, 4, 4, 4]}
        )
        # 使用 crosstab 函数计算 df 的交叉表，设置 margins=True 和 dropna=True
        actual = crosstab(df.a, df.b, margins=True, dropna=True)
        # 创建一个包含预期结果的 DataFrame 对象 expected
        expected = DataFrame([[1, 0, 1], [0, 1, 1], [1, 1, 2]])
        expected.index = Index([1.0, 2.0, "All"], name="a")
        expected.columns = Index([3.0, 4.0, "All"], name="b")
        # 断言计算结果 actual 与预期结果 expected 相等
        tm.assert_frame_equal(actual, expected)

    def test_margin_dropna3(self):
        # 创建包含字典数据的 Pandas DataFrame 对象 df
        # DataFrame 包含两列 'a' 和 'b'，其中 'a' 列包含数值和 NaN，'b' 列包含整数数据
        df = DataFrame(
            {"a": [1, np.nan, np.nan, np.nan, np.nan, 2], "b": [3, 3, 4, 4, 4, 4]}
        )
        # 使用 crosstab 函数计算 df 的交叉表，设置 margins=True 和 dropna=True
        actual = crosstab(df.a, df.b, margins=True, dropna=True)
        # 创建一个包含预期结果的 DataFrame 对象 expected
        expected = DataFrame([[1, 0, 1], [0, 1, 1], [1, 1, 2]])
        expected.index = Index([1.0, 2.0, "All"], name="a")
        expected.columns = Index([3, 4, "All"], name="b")
        # 断言计算结果 actual 与预期结果 expected 相等
        tm.assert_frame_equal(actual, expected)
    def test_margin_dropna4(self):
        # GH 12642
        # _add_margins raises KeyError: Level None not found
        # 当 margins=True 且 dropna=False 时，_add_margins 抛出 KeyError: Level None not found
        # GH: 10772: Keep np.nan in result with dropna=False
        # GH: 10772: 当 dropna=False 时，结果中保留 np.nan
        # 创建包含指定数据的 DataFrame 对象
        df = DataFrame({"a": [1, 2, 2, 2, 2, np.nan], "b": [3, 3, 4, 4, 4, 4]})
        # 调用 crosstab 函数生成交叉表，设定 margins=True 和 dropna=False
        actual = crosstab(df.a, df.b, margins=True, dropna=False)
        # 创建预期的 DataFrame 对象
        expected = DataFrame([[1, 0, 1.0], [1, 3, 4.0], [0, 1, np.nan], [2, 4, 6.0]])
        # 设置预期 DataFrame 的行索引
        expected.index = Index([1.0, 2.0, np.nan, "All"], name="a")
        # 设置预期 DataFrame 的列索引
        expected.columns = Index([3, 4, "All"], name="b")
        # 使用 assert_frame_equal 函数比较实际结果和预期结果是否相等
        tm.assert_frame_equal(actual, expected)

    def test_margin_dropna5(self):
        # GH: 10772: Keep np.nan in result with dropna=False
        # GH: 10772: 当 dropna=False 时，结果中保留 np.nan
        # 创建包含指定数据的 DataFrame 对象
        df = DataFrame(
            {"a": [1, np.nan, np.nan, np.nan, 2, np.nan], "b": [3, np.nan, 4, 4, 4, 4]}
        )
        # 调用 crosstab 函数生成交叉表，设定 margins=True 和 dropna=False
        actual = crosstab(df.a, df.b, margins=True, dropna=False)
        # 创建预期的 DataFrame 对象
        expected = DataFrame(
            [[1, 0, 0, 1.0], [0, 1, 0, 1.0], [0, 3, 1, np.nan], [1, 4, 0, 6.0]]
        )
        # 设置预期 DataFrame 的行索引
        expected.index = Index([1.0, 2.0, np.nan, "All"], name="a")
        # 设置预期 DataFrame 的列索引
        expected.columns = Index([3.0, 4.0, np.nan, "All"], name="b")
        # 使用 assert_frame_equal 函数比较实际结果和预期结果是否相等
        tm.assert_frame_equal(actual, expected)
    # 定义测试方法 test_margin_dropna6，用于验证 GH-10772: 当 dropna=False 时保留 np.nan 在结果中
    def test_margin_dropna6(self):
        # 创建包含字符串的 numpy 数组 a
        a = np.array(["foo", "foo", "foo", "bar", "bar", "foo", "foo"], dtype=object)
        # 创建包含字符串和 np.nan 的 numpy 数组 b
        b = np.array(["one", "one", "two", "one", "two", np.nan, "two"], dtype=object)
        # 创建包含字符串的 numpy 数组 c
        c = np.array(
            ["dull", "dull", "dull", "dull", "dull", "shiny", "shiny"], dtype=object
        )

        # 使用 crosstab 函数生成实际结果 actual，设置 margins=True 和 dropna=False
        actual = crosstab(
            a, [b, c], rownames=["a"], colnames=["b", "c"], margins=True, dropna=False
        )
        # 使用 MultiIndex.from_arrays 创建预期的列索引 m
        m = MultiIndex.from_arrays(
            [
                ["one", "one", "two", "two", np.nan, np.nan, "All"],
                ["dull", "shiny", "dull", "shiny", "dull", "shiny", ""],
            ],
            names=["b", "c"],
        )
        # 创建预期的 DataFrame expected
        expected = DataFrame(
            [[1, 0, 1, 0, 0, 0, 2], [2, 0, 1, 1, 0, 1, 5], [3, 0, 2, 1, 0, 0, 7]],
            columns=m,
        )
        # 设置预期 DataFrame 的行索引
        expected.index = Index(["bar", "foo", "All"], name="a")
        # 使用 tm.assert_frame_equal 检验 actual 和 expected 是否相等

        tm.assert_frame_equal(actual, expected)

        # 使用 crosstab 函数生成实际结果 actual，设置 margins=True 和 dropna=False
        actual = crosstab(
            [a, b], c, rownames=["a", "b"], colnames=["c"], margins=True, dropna=False
        )
        # 使用 MultiIndex.from_arrays 创建预期的行索引 m
        m = MultiIndex.from_arrays(
            [
                ["bar", "bar", "bar", "foo", "foo", "foo", "All"],
                ["one", "two", np.nan, "one", "two", np.nan, ""],
            ],
            names=["a", "b"],
        )
        # 创建预期的 DataFrame expected
        expected = DataFrame(
            [
                [1, 0, 1.0],
                [1, 0, 1.0],
                [0, 0, np.nan],
                [2, 0, 2.0],
                [1, 1, 2.0],
                [0, 1, np.nan],
                [5, 2, 7.0],
            ],
            index=m,
        )
        # 设置预期 DataFrame 的列索引
        expected.columns = Index(["dull", "shiny", "All"], name="c")
        # 使用 tm.assert_frame_equal 检验 actual 和 expected 是否相等

        tm.assert_frame_equal(actual, expected)

        # 使用 crosstab 函数生成实际结果 actual，设置 margins=True 和 dropna=True
        actual = crosstab(
            [a, b], c, rownames=["a", "b"], colnames=["c"], margins=True, dropna=True
        )
        # 使用 MultiIndex.from_arrays 创建预期的行索引 m
        m = MultiIndex.from_arrays(
            [["bar", "bar", "foo", "foo", "All"], ["one", "two", "one", "two", ""]],
            names=["a", "b"],
        )
        # 创建预期的 DataFrame expected
        expected = DataFrame(
            [[1, 0, 1], [1, 0, 1], [2, 0, 2], [1, 1, 2], [5, 1, 6]], index=m
        )
        # 设置预期 DataFrame 的列索引
        expected.columns = Index(["dull", "shiny", "All"], name="c")
        # 使用 tm.assert_frame_equal 检验 actual 和 expected 是否相等
    # 定义测试方法，用于验证交叉表格的归一化功能
    def test_crosstab_normalize(self):
        # 创建一个包含三列的数据帧，包括整数和NaN值
        df = DataFrame(
            {"a": [1, 2, 2, 2, 2], "b": [3, 3, 4, 4, 4], "c": [1, 1, np.nan, 1, 1]}
        )

        # 创建行索引，表示"a"列中的唯一值
        rindex = Index([1, 2], name="a")
        # 创建列索引，表示"b"列中的唯一值
        cindex = Index([3, 4], name="b")

        # 创建完全归一化的数据帧，用于全局归一化的验证
        full_normal = DataFrame([[0.2, 0], [0.2, 0.6]], index=rindex, columns=cindex)
        # 创建行归一化的数据帧，用于行归一化的验证
        row_normal = DataFrame([[1.0, 0], [0.25, 0.75]], index=rindex, columns=cindex)
        # 创建列归一化的数据帧，用于列归一化的验证
        col_normal = DataFrame([[0.5, 0], [0.5, 1.0]], index=rindex, columns=cindex)

        # 检查所有归一化参数是否正常
        tm.assert_frame_equal(crosstab(df.a, df.b, normalize="all"), full_normal)
        tm.assert_frame_equal(crosstab(df.a, df.b, normalize=True), full_normal)
        tm.assert_frame_equal(crosstab(df.a, df.b, normalize="index"), row_normal)
        tm.assert_frame_equal(crosstab(df.a, df.b, normalize="columns"), col_normal)
        tm.assert_frame_equal(
            crosstab(df.a, df.b, normalize=1),
            crosstab(df.a, df.b, normalize="columns"),
        )
        tm.assert_frame_equal(
            crosstab(df.a, df.b, normalize=0), crosstab(df.a, df.b, normalize="index")
        )

        # 创建包含行边缘和列边缘的行归一化数据帧，用于边缘归一化的验证
        row_normal_margins = DataFrame(
            [[1.0, 0], [0.25, 0.75], [0.4, 0.6]],
            index=Index([1, 2, "All"], name="a", dtype="object"),
            columns=Index([3, 4], name="b", dtype="object"),
        )
        # 创建包含行边缘和列边缘的列归一化数据帧，用于边缘归一化的验证
        col_normal_margins = DataFrame(
            [[0.5, 0, 0.2], [0.5, 1.0, 0.8]],
            index=Index([1, 2], name="a", dtype="object"),
            columns=Index([3, 4, "All"], name="b", dtype="object"),
        )
        # 创建包含所有边缘的完全归一化数据帧，用于边缘归一化的验证
        all_normal_margins = DataFrame(
            [[0.2, 0, 0.2], [0.2, 0.6, 0.8], [0.4, 0.6, 1]],
            index=Index([1, 2, "All"], name="a", dtype="object"),
            columns=Index([3, 4, "All"], name="b", dtype="object"),
        )
        # 验证带有行和列边缘的行归一化
        tm.assert_frame_equal(
            crosstab(df.a, df.b, normalize="index", margins=True), row_normal_margins
        )
        # 验证带有行和列边缘的列归一化
        tm.assert_frame_equal(
            crosstab(df.a, df.b, normalize="columns", margins=True), col_normal_margins
        )
        # 验证带有所有边缘的全局归一化
        tm.assert_frame_equal(
            crosstab(df.a, df.b, normalize=True, margins=True), all_normal_margins
        )
    # 定义测试函数，用于测试带有标准化和空值处理的交叉制表功能
    def test_crosstab_normalize_arrays(self):
        # GH#12578：GitHub issue编号，指明此测试的背景或相关问题

        # 创建一个包含列"a", "b", "c"的DataFrame对象
        df = DataFrame(
            {"a": [1, 2, 2, 2, 2], "b": [3, 3, 4, 4, 4], "c": [1, 1, np.nan, 1, 1]}
        )

        # 调用crosstab函数，测试输入为两个numpy数组和一个numpy数组
        crosstab(
            [np.array([1, 1, 2, 2]), np.array([1, 2, 1, 2])], np.array([1, 2, 1, 2])
        )

        # 创建一个包含标准化后计数的DataFrame对象
        norm_counts = DataFrame(
            [[0.25, 0, 0.25], [0.25, 0.5, 0.75], [0.5, 0.5, 1]],
            index=Index([1, 2, "All"], name="a", dtype="object"),
            columns=Index([3, 4, "All"], name="b"),
        )

        # 调用crosstab函数，测试输入为DataFrame的列'a', 'b', 'c'，使用aggfunc="count"和normalize="all"
        test_case = crosstab(
            df.a, df.b, df.c, aggfunc="count", normalize="all", margins=True
        )

        # 断言测试结果与标准化后计数的DataFrame对象一致
        tm.assert_frame_equal(test_case, norm_counts)

        # 更新DataFrame对象的内容，修改列'c'的值
        df = DataFrame(
            {"a": [1, 2, 2, 2, 2], "b": [3, 3, 4, 4, 4], "c": [0, 4, np.nan, 3, 3]}
        )

        # 创建一个包含标准化后求和的DataFrame对象
        norm_sum = DataFrame(
            [[0, 0, 0.0], [0.4, 0.6, 1], [0.4, 0.6, 1]],
            index=Index([1, 2, "All"], name="a", dtype="object"),
            columns=Index([3, 4, "All"], name="b", dtype="object"),
        )

        # 调用crosstab函数，测试输入为DataFrame的列'a', 'b', 'c'，使用aggfunc=np.sum和normalize="all"
        test_case = crosstab(
            df.a, df.b, df.c, aggfunc=np.sum, normalize="all", margins=True
        )

        # 断言测试结果与标准化后求和的DataFrame对象一致
        tm.assert_frame_equal(test_case, norm_sum)

    # 定义测试函数，用于测试处理空值情况下的交叉制表功能
    def test_crosstab_with_empties(self):
        # 创建一个包含列"a", "b", "c"的DataFrame对象，其中所有值为NaN
        df = DataFrame(
            {
                "a": [1, 2, 2, 2, 2],
                "b": [3, 3, 4, 4, 4],
                "c": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )

        # 创建一个空DataFrame对象，所有值为0
        empty = DataFrame(
            [[0.0, 0.0], [0.0, 0.0]],
            index=Index([1, 2], name="a", dtype="int64"),
            columns=Index([3, 4], name="b"),
        )

        # 针对不同的normalize参数值进行循环测试
        for i in [True, "index", "columns"]:
            # 调用crosstab函数，测试输入为DataFrame的列'a', 'b', 'c'，aggfunc="count"和不同的normalize参数
            calculated = crosstab(df.a, df.b, values=df.c, aggfunc="count", normalize=i)
            # 断言测试结果与空DataFrame对象一致
            tm.assert_frame_equal(empty, calculated)

        # 创建一个包含NaN值的DataFrame对象
        nans = DataFrame(
            [[0.0, np.nan], [0.0, 0.0]],
            index=Index([1, 2], name="a", dtype="int64"),
            columns=Index([3, 4], name="b"),
        )

        # 调用crosstab函数，测试输入为DataFrame的列'a', 'b', 'c'，aggfunc="count"和normalize=False
        calculated = crosstab(df.a, df.b, values=df.c, aggfunc="count", normalize=False)
        # 断言测试结果与包含NaN值的DataFrame对象一致
        tm.assert_frame_equal(nans, calculated)
    def test_crosstab_errors(self):
        # 测试函数：test_crosstab_errors
        # Issue 12578

        # 创建一个包含三列的数据框 DataFrame
        df = DataFrame(
            {"a": [1, 2, 2, 2, 2], "b": [3, 3, 4, 4, 4], "c": [1, 1, np.nan, 1, 1]}
        )

        # 设置错误消息字符串
        error = "values cannot be used without an aggfunc."
        # 使用 pytest 的上下文管理器检查是否引发 ValueError 异常，匹配指定错误消息
        with pytest.raises(ValueError, match=error):
            crosstab(df.a, df.b, values=df.c)

        # 设置错误消息字符串
        error = "aggfunc cannot be used without values"
        # 使用 pytest 的上下文管理器检查是否引发 ValueError 异常，匹配指定错误消息
        with pytest.raises(ValueError, match=error):
            crosstab(df.a, df.b, aggfunc=np.mean)

        # 设置错误消息字符串
        error = "Not a valid normalize argument"
        # 使用 pytest 的上下文管理器检查是否引发 ValueError 异常，匹配指定错误消息
        with pytest.raises(ValueError, match=error):
            crosstab(df.a, df.b, normalize="42")

        # 使用 pytest 的上下文管理器检查是否引发 ValueError 异常，匹配相同的错误消息
        with pytest.raises(ValueError, match=error):
            crosstab(df.a, df.b, normalize=42)

        # 设置错误消息字符串
        error = "Not a valid margins argument"
        # 使用 pytest 的上下文管理器检查是否引发 ValueError 异常，匹配指定错误消息
        with pytest.raises(ValueError, match=error):
            crosstab(df.a, df.b, normalize="all", margins=42)

    def test_crosstab_with_categorial_columns(self):
        # 测试函数：test_crosstab_with_categorial_columns
        # GH 8860

        # 创建一个包含两列的数据框 DataFrame
        df = DataFrame(
            {
                "MAKE": ["Honda", "Acura", "Tesla", "Honda", "Honda", "Acura"],
                "MODEL": ["Sedan", "Sedan", "Electric", "Pickup", "Sedan", "Sedan"],
            }
        )
        # 指定分类变量的类别，并将 "MODEL" 列转换为分类类型
        categories = ["Sedan", "Electric", "Pickup"]
        df["MODEL"] = df["MODEL"].astype("category").cat.set_categories(categories)
        
        # 执行 crosstab 操作，生成结果
        result = crosstab(df["MAKE"], df["MODEL"])

        # 期望的行索引
        expected_index = Index(["Acura", "Honda", "Tesla"], name="MAKE")
        # 期望的列索引，使用 CategoricalIndex 表示分类变量
        expected_columns = CategoricalIndex(
            categories, categories=categories, ordered=False, name="MODEL"
        )
        # 期望的数据框结果
        expected_data = [[2, 0, 0], [2, 0, 1], [0, 1, 0]]
        expected = DataFrame(
            expected_data, index=expected_index, columns=expected_columns
        )
        
        # 使用 assert_frame_equal 函数比较结果和期望值，确认测试通过
        tm.assert_frame_equal(result, expected)
    def test_crosstab_with_numpy_size(self):
        # 测试用例：使用 numpy 的 size 函数进行交叉表计算
        # GH 4003：GitHub issue 编号
        df = DataFrame(
            {
                "A": ["one", "one", "two", "three"] * 6,
                "B": ["A", "B", "C"] * 8,
                "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
                "D": np.random.default_rng(2).standard_normal(24),
                "E": np.random.default_rng(2).standard_normal(24),
            }
        )
        # 调用 crosstab 函数生成交叉表
        result = crosstab(
            index=[df["A"], df["B"]],
            columns=[df["C"]],
            margins=True,
            aggfunc=np.size,  # 指定聚合函数为 numpy 的 size 函数
            values=df["D"],
        )
        # 期望的行索引
        expected_index = MultiIndex(
            levels=[["All", "one", "three", "two"], ["", "A", "B", "C"]],
            codes=[[1, 1, 1, 2, 2, 2, 3, 3, 3, 0], [1, 2, 3, 1, 2, 3, 1, 2, 3, 0]],
            names=["A", "B"],
        )
        # 期望的列索引
        expected_column = Index(["bar", "foo", "All"], name="C")
        # 期望的数据数组
        expected_data = np.array(
            [
                [2.0, 2.0, 4.0],
                [2.0, 2.0, 4.0],
                [2.0, 2.0, 4.0],
                [2.0, np.nan, 2.0],
                [np.nan, 2.0, 2.0],
                [2.0, np.nan, 2.0],
                [np.nan, 2.0, 2.0],
                [2.0, np.nan, 2.0],
                [np.nan, 2.0, 2.0],
                [12.0, 12.0, 24.0],
            ]
        )
        # 构建期望的 DataFrame
        expected = DataFrame(
            expected_data, index=expected_index, columns=expected_column
        )
        # 调整 "All" 列的数据类型为 int64
        expected["All"] = expected["All"].astype("int64")
        # 使用 assert_frame_equal 函数比较实际结果和期望结果
        tm.assert_frame_equal(result, expected)

    def test_crosstab_duplicate_names(self):
        # 测试用例：处理重复的标签名称
        # GH 13279 / 22529：GitHub issue 编号

        s1 = Series(range(3), name="foo")
        s2_foo = Series(range(1, 4), name="foo")
        s2_bar = Series(range(1, 4), name="bar")
        s3 = Series(range(3), name="waldo")

        # 使用重复的标签计算结果，并通过重命名来检查
        mapper = {"bar": "foo"}

        # 测试：重复的行和列标签
        result = crosstab(s1, s2_foo)
        expected = crosstab(s1, s2_bar).rename_axis(columns=mapper, axis=1)
        tm.assert_frame_equal(result, expected)

        # 测试：重复的行标签和唯一的列标签
        result = crosstab([s1, s2_foo], s3)
        expected = crosstab([s1, s2_bar], s3).rename_axis(index=mapper, axis=0)
        tm.assert_frame_equal(result, expected)

        # 测试：唯一的行标签和重复的列标签
        result = crosstab(s3, [s1, s2_foo])
        expected = crosstab(s3, [s1, s2_bar]).rename_axis(columns=mapper, axis=1)

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("names", [["a", ("b", "c")], [("a", "b"), "c"]])
    def test_crosstab_tuple_name(self, names):
        # 创建两个Series对象，分别使用传入的names列表中的第一个和第二个元素作为名称
        s1 = Series(range(3), name=names[0])
        s2 = Series(range(1, 4), name=names[1])

        # 使用MultiIndex.from_arrays方法创建一个多级索引对象mi，
        # 该索引对象的级别名称分别使用传入的names列表中的第一个和第二个元素
        mi = MultiIndex.from_arrays([range(3), range(1, 4)], names=names)
        
        # 创建一个预期的DataFrame对象，使用mi作为索引，填充值为1，列进行透视操作
        expected = Series(1, index=mi).unstack(1, fill_value=0)

        # 调用crosstab函数，传入s1和s2，生成结果DataFrame对象result
        result = crosstab(s1, s2)
        
        # 使用tm.assert_frame_equal函数比较result和expected，确保它们相等
        tm.assert_frame_equal(result, expected)

    def test_crosstab_both_tuple_names(self):
        # GH 18321
        # 创建两个Series对象，分别使用元组("a", "b")和("c", "d")作为名称
        s1 = Series(range(3), name=("a", "b"))
        s2 = Series(range(3), name=("c", "d"))

        # 创建预期的DataFrame对象expected，使用np.eye生成对角矩阵，数据类型为int64，
        # 索引和列分别使用元组("a", "b")和("c", "d")作为名称
        expected = DataFrame(
            np.eye(3, dtype="int64"),
            index=Index(range(3), name=("a", "b")),
            columns=Index(range(3), name=("c", "d")),
        )

        # 调用crosstab函数，传入s1和s2，生成结果DataFrame对象result
        result = crosstab(s1, s2)
        
        # 使用tm.assert_frame_equal函数比较result和expected，确保它们相等
        tm.assert_frame_equal(result, expected)

    def test_crosstab_unsorted_order(self):
        # 创建一个DataFrame对象df，包含两列'b'和'a'，索引为["C", "A", "B"]
        df = DataFrame({"b": [3, 1, 2], "a": [5, 4, 6]}, index=["C", "A", "B"])

        # 调用crosstab函数，传入df.index和[df.b, df.a]，生成结果DataFrame对象result
        result = crosstab(df.index, [df.b, df.a])
        
        # 创建预期的DataFrame对象expected，索引为Index(["A", "B", "C"], name="row_0")，
        # 列为MultiIndex对象，每个元组分别使用("b", "a")作为名称
        e_idx = Index(["A", "B", "C"], name="row_0")
        e_columns = MultiIndex.from_tuples([(1, 4), (2, 6), (3, 5)], names=["b", "a"])
        expected = DataFrame(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], index=e_idx, columns=e_columns
        )
        
        # 使用tm.assert_frame_equal函数比较result和expected，确保它们相等
        tm.assert_frame_equal(result, expected)

    def test_crosstab_normalize_multiple_columns(self):
        # GH 15150
        # 创建一个DataFrame对象df，包含五列'A'、'B'、'C'、'D'、'E'
        df = DataFrame(
            {
                "A": ["one", "one", "two", "three"] * 6,
                "B": ["A", "B", "C"] * 8,
                "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
                "D": [0] * 24,
                "E": [0] * 24,
            }
        )

        # 调用crosstab函数，传入[df.A, df.B]、df.C、df.D等参数，生成结果DataFrame对象result
        result = crosstab(
            [df.A, df.B],
            df.C,
            values=df.D,
            aggfunc=np.sum,
            normalize=True,
            margins=True,
        )

        # 创建预期的DataFrame对象expected，数据为数组[0] * 29 + [1]，形状为(10, 3)，
        # 列使用Index(["bar", "foo", "All"], name="C")，索引使用MultiIndex对象，
        # 每个元组分别使用("A", "B")作为名称
        expected = DataFrame(
            np.array([0] * 29 + [1], dtype=float).reshape(10, 3),
            columns=Index(["bar", "foo", "All"], name="C"),
            index=MultiIndex.from_tuples(
                [
                    ("one", "A"),
                    ("one", "B"),
                    ("one", "C"),
                    ("three", "A"),
                    ("three", "B"),
                    ("three", "C"),
                    ("two", "A"),
                    ("two", "B"),
                    ("two", "C"),
                    ("All", ""),
                ],
                names=["A", "B"],
            ),
        )
        
        # 使用tm.assert_frame_equal函数比较result和expected，确保它们相等
        tm.assert_frame_equal(result, expected)
    def test_margin_normalize(self):
        # 测试用例 GH 27500

        # 创建一个 DataFrame 对象 df，包含五列数据
        df = DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                ],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
            }
        )

        # 对索引进行标准化处理
        result = crosstab(
            [df.A, df.B], df.C, margins=True, margins_name="Sub-Total", normalize=0
        )

        # 创建预期结果 DataFrame 对象 expected
        expected = DataFrame(
            [[0.5, 0.5], [0.5, 0.5], [0.666667, 0.333333], [0, 1], [0.444444, 0.555556]]
        )

        # 设置预期结果的索引为多级索引 MultiIndex
        expected.index = MultiIndex(
            levels=[["Sub-Total", "bar", "foo"], ["", "one", "two"]],
            codes=[[1, 1, 2, 2, 0], [1, 2, 1, 2, 0]],
            names=["A", "B"],
        )

        # 设置预期结果的列名为 Index 对象
        expected.columns = Index(["large", "small"], name="C")

        # 使用测试工具函数检查结果是否符合预期
        tm.assert_frame_equal(result, expected)

        # 对列进行标准化处理
        result = crosstab(
            [df.A, df.B], df.C, margins=True, margins_name="Sub-Total", normalize=1
        )

        # 创建新的预期结果 DataFrame 对象 expected
        expected = DataFrame(
            [
                [0.25, 0.2, 0.222222],
                [0.25, 0.2, 0.222222],
                [0.5, 0.2, 0.333333],
                [0, 0.4, 0.222222],
            ]
        )

        # 设置预期结果的列名为 Index 对象
        expected.columns = Index(["large", "small", "Sub-Total"], name="C")

        # 设置预期结果的索引为多级索引 MultiIndex
        expected.index = MultiIndex(
            levels=[["bar", "foo"], ["one", "two"]],
            codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
            names=["A", "B"],
        )

        # 使用测试工具函数检查结果是否符合预期
        tm.assert_frame_equal(result, expected)

        # 对索引和列同时进行标准化处理
        result = crosstab(
            [df.A, df.B], df.C, margins=True, margins_name="Sub-Total", normalize=True
        )

        # 创建新的预期结果 DataFrame 对象 expected
        expected = DataFrame(
            [
                [0.111111, 0.111111, 0.222222],
                [0.111111, 0.111111, 0.222222],
                [0.222222, 0.111111, 0.333333],
                [0.000000, 0.222222, 0.222222],
                [0.444444, 0.555555, 1],
            ]
        )

        # 设置预期结果的列名为 Index 对象
        expected.columns = Index(["large", "small", "Sub-Total"], name="C")

        # 设置预期结果的索引为多级索引 MultiIndex
        expected.index = MultiIndex(
            levels=[["Sub-Total", "bar", "foo"], ["", "one", "two"]],
            codes=[[1, 1, 2, 2, 0], [1, 2, 1, 2, 0]],
            names=["A", "B"],
        )

        # 使用测试工具函数检查结果是否符合预期
        tm.assert_frame_equal(result, expected)
    def test_margin_normalize_multiple_columns(self):
        # GH 35144
        # GH issue 35144: Test case for using multiple columns with margins and normalization
        # 创建包含多列数据的DataFrame
        df = DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                ],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
            }
        )
        # 使用crosstab函数生成交叉表，包含索引、列、边界值、归一化等参数
        result = crosstab(
            index=df.C,
            columns=[df.A, df.B],
            margins=True,
            margins_name="margin",
            normalize=True,
        )
        # 预期结果DataFrame，包含归一化后的数据
        expected = DataFrame(
            [
                [0.111111, 0.111111, 0.222222, 0.000000, 0.444444],
                [0.111111, 0.111111, 0.111111, 0.222222, 0.555556],
                [0.222222, 0.222222, 0.333333, 0.222222, 1.0],
            ],
            index=["large", "small", "margin"],
        )
        # 设置预期DataFrame的列索引
        expected.columns = MultiIndex(
            levels=[["bar", "foo", "margin"], ["", "one", "two"]],
            codes=[[0, 0, 1, 1, 2], [1, 2, 1, 2, 0]],
            names=["A", "B"],
        )
        # 设置预期DataFrame的行索引名称
        expected.index.name = "C"
        # 使用tm.assert_frame_equal函数比较实际结果与预期结果DataFrame
        tm.assert_frame_equal(result, expected)

    def test_margin_support_Float(self):
        # GH 50313
        # GH issue 50313: Test case for using Float64 formats and aggfunc with margins
        # 创建包含Float64格式数据的DataFrame
        df = DataFrame(
            {"A": [1, 2, 2, 1], "B": [3, 3, 4, 5], "C": [-1.0, 10.0, 1.0, 10.0]},
            dtype="Float64",
        )
        # 使用crosstab函数生成交叉表，包含值、聚合函数、边界值等参数
        result = crosstab(
            df["A"],
            df["B"],
            values=df["C"],
            aggfunc="sum",
            margins=True,
        )
        # 预期结果DataFrame，包含聚合后的数据和边界值
        expected = DataFrame(
            [
                [-1.0, pd.NA, 10.0, 9.0],
                [10.0, 1.0, pd.NA, 11.0],
                [9.0, 1.0, 10.0, 20.0],
            ],
            index=Index([1.0, 2.0, "All"], dtype="object", name="A"),
            columns=Index([3.0, 4.0, 5.0, "All"], dtype="object", name="B"),
            dtype="Float64",
        )
        # 使用tm.assert_frame_equal函数比较实际结果与预期结果DataFrame
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于测试带有有序分类列的边际情况
    def test_margin_with_ordered_categorical_column(self):
        # GH 25278：GitHub 上的 issue 编号
        # 创建一个 DataFrame，包含两列："First" 和 "Second"
        df = DataFrame(
            {
                "First": ["B", "B", "C", "A", "B", "C"],
                "Second": ["C", "B", "B", "B", "C", "A"],
            }
        )
        # 将 "First" 列的数据类型转换为有序分类类型
        df["First"] = df["First"].astype(CategoricalDtype(ordered=True))
        # 指定自定义的分类顺序
        customized_categories_order = ["C", "A", "B"]
        # 按照自定义顺序重新排列 "First" 列的分类
        df["First"] = df["First"].cat.reorder_categories(customized_categories_order)
        # 计算 "First" 列和 "Second" 列的交叉表，包括边际汇总
        result = crosstab(df["First"], df["Second"], margins=True)

        # 期望的行索引，包括 "All" 边际汇总
        expected_index = Index(["C", "A", "B", "All"], name="First")
        # 期望的列索引，包括 "All" 边际汇总
        expected_columns = Index(["A", "B", "C", "All"], name="Second")
        # 期望的数据表格，展示了交叉表的预期结果
        expected_data = [[1, 1, 0, 2], [0, 1, 0, 1], [0, 1, 2, 3], [1, 3, 2, 6]]
        # 创建期望的 DataFrame，用于与计算结果进行比较
        expected = DataFrame(
            expected_data, index=expected_index, columns=expected_columns
        )
        # 使用测试框架中的方法，比较计算结果和期望结果是否相等
        tm.assert_frame_equal(result, expected)
# 使用参数化测试框架，对两个变量进行多组参数化测试
@pytest.mark.parametrize("a_dtype", ["category", "int64"])
@pytest.mark.parametrize("b_dtype", ["category", "int64"])
def test_categoricals(a_dtype, b_dtype):
    # 创建随机数生成器对象
    g = np.random.default_rng(2)
    # 生成一个长度为100的随机整数序列，并将其转换为指定的数据类型a_dtype
    a = Series(g.integers(0, 3, size=100)).astype(a_dtype)
    # 生成一个长度为100的随机整数序列，并将其转换为指定的数据类型b_dtype
    b = Series(g.integers(0, 2, size=100)).astype(b_dtype)
    # 计算a和b的交叉列联表，包括边际和缺失值不丢弃
    result = crosstab(a, b, margins=True, dropna=False)
    # 创建列索引对象，包括0, 1, "All"三个值，数据类型为object
    columns = Index([0, 1, "All"], dtype="object", name="col_0")
    # 创建行索引对象，包括0, 1, 2, "All"四个值，数据类型为object
    index = Index([0, 1, 2, "All"], dtype="object", name="row_0")
    # 创建预期的数据框，数据内容为二维列表values，行索引为index，列索引为columns
    values = [[10, 18, 28], [23, 16, 39], [17, 16, 33], [50, 50, 100]]
    expected = DataFrame(values, index, columns)
    # 使用测试工具函数验证计算结果与预期结果是否一致
    tm.assert_frame_equal(result, expected)

    # 验证当分类变量a中不包含所有可能的值时的情况
    # 将a中等于1的值修改为2，以模拟不包含所有值的情况
    a.loc[a == 1] = 2
    # 检查a的数据类型是否为CategoricalDtype
    a_is_cat = isinstance(a.dtype, CategoricalDtype)
    # 断言如果a不是分类变量，或者值为1的计数为0
    assert not a_is_cat or a.value_counts().loc[1] == 0
    # 重新计算a和b的交叉列联表，包括边际和缺失值不丢弃
    result = crosstab(a, b, margins=True, dropna=False)
    # 创建预期的数据框，数据内容为二维列表values，行索引为index，列索引为columns
    values = [[10, 18, 28], [0, 0, 0], [40, 32, 72], [50, 50, 100]]
    expected = DataFrame(values, index, columns)
    # 如果a不是分类变量，则修改预期结果的格式，只保留行索引为0, 2, "All"的数据，并将"All"列的数据类型转换为int64
    if not a_is_cat:
        expected = expected.loc[[0, 2, "All"]]
        expected["All"] = expected["All"].astype("int64")
    # 使用测试工具函数验证计算结果与预期结果是否一致
    tm.assert_frame_equal(result, expected)
```