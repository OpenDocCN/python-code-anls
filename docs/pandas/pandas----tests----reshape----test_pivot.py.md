# `D:\src\scipysrc\pandas\pandas\tests\reshape\test_pivot.py`

```
    # 从 datetime 模块导入 date, datetime, timedelta 类
    # 从 itertools 模块导入 product 函数
    import re  # 导入 re 模块，用于正则表达式操作

    import numpy as np  # 导入 NumPy 库，并使用 np 别名
    import pytest  # 导入 pytest 库

    # 从 pandas._config 模块导入 using_pyarrow_string_dtype 变量
    from pandas._config import using_pyarrow_string_dtype

    # 从 pandas.compat.numpy 模块导入 np_version_gte1p25 函数
    from pandas.compat.numpy import np_version_gte1p25

    import pandas as pd  # 导入 Pandas 库，并使用 pd 别名
    # 从 pandas 模块导入多个类和函数，包括 Categorical, DataFrame, Grouper, Index, MultiIndex, Series, concat, date_range 等
    from pandas import (
        Categorical,
        DataFrame,
        Grouper,
        Index,
        MultiIndex,
        Series,
        concat,
        date_range,
    )

    # 从 pandas._testing 模块导入 tm 对象
    import pandas._testing as tm
    # 从 pandas.api.types 模块导入 CategoricalDtype 类型
    from pandas.api.types import CategoricalDtype
    # 从 pandas.core.reshape 模块导入 reshape 函数，并使用 reshape_lib 别名
    from pandas.core.reshape import reshape as reshape_lib
    # 从 pandas.core.reshape.pivot 模块导入 pivot_table 函数

    from pandas.core.reshape.pivot import pivot_table


    class TestPivotTable:
        # 为测试数据创建 pytest fixture
        @pytest.fixture
        def data(self):
            return DataFrame(
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
                    # 使用随机数生成数据列 D, E, F
                    "D": np.random.default_rng(2).standard_normal(11),
                    "E": np.random.default_rng(2).standard_normal(11),
                    "F": np.random.default_rng(2).standard_normal(11),
                }
            )

        # 定义测试方法 test_pivot_table，接受 observed 和 data 两个参数
        def test_pivot_table(self, observed, data):
            index = ["A", "B"]  # 定义索引列为 A 和 B
            columns = "C"  # 定义列名为 C
            # 调用 pivot_table 函数生成数据透视表
            table = pivot_table(
                data, values="D", index=index, columns=columns, observed=observed
            )

            # 使用 DataFrame 对象的 pivot_table 方法生成数据透视表
            table2 = data.pivot_table(
                values="D", index=index, columns=columns, observed=observed
            )
            # 使用 pandas._testing 模块中的 assert_frame_equal 方法比较两个数据透视表的内容
            tm.assert_frame_equal(table, table2)

            # 调用 pivot_table 函数，生成数据透视表，没有将其赋值给变量
            pivot_table(data, values="D", index=index, observed=observed)

            # 如果索引列的长度大于 1，则断言表格的索引名称与元组 index 相同
            if len(index) > 1:
                assert table.index.names == tuple(index)
            else:
                # 否则，断言表格的索引名称与 index 列表的第一个元素相同
                assert table.index.name == index[0]

            # 如果列的长度大于 1，则断言表格的列名称与 columns 相同
            if len(columns) > 1:
                assert table.columns.names == columns
            else:
                # 否则，断言表格的列名称与 columns 列表的第一个元素相同
                assert table.columns.name == columns[0]

            # 生成预期结果，通过对数据进行分组并计算均值，然后使用 unstack 方法重塑数据
            expected = data.groupby(index + [columns])["D"].agg("mean").unstack()
            # 使用 pandas._testing 模块中的 assert_frame_equal 方法比较生成的表格与预期结果的内容
            tm.assert_frame_equal(table, expected)
    # 定义一个测试函数，用于验证在观察到的情况下，数据透视表的行为是否符合预期
    def test_pivot_table_categorical_observed_equal(self, observed):
        # 创建一个包含列 'col1', 'col2', 'col3' 的数据框
        df = DataFrame(
            {"col1": list("abcde"), "col2": list("fghij"), "col3": [1, 2, 3, 4, 5]}
        )

        # 根据 'col1' 和 'col2' 对 'col3' 进行聚合求和，创建预期的透视表
        expected = df.pivot_table(
            index="col1", values="col3", columns="col2", aggfunc="sum", fill_value=0
        )

        # 将预期的透视表的行和列转换为分类数据类型
        expected.index = expected.index.astype("category")
        expected.columns = expected.columns.astype("category")

        # 将数据框中的 'col1' 和 'col2' 列转换为分类数据类型
        df.col1 = df.col1.astype("category")
        df.col2 = df.col2.astype("category")

        # 根据 'col1' 和 'col2' 对 'col3' 进行聚合求和，生成实际的透视表
        result = df.pivot_table(
            index="col1",
            values="col3",
            columns="col2",
            aggfunc="sum",
            fill_value=0,
            observed=observed,
        )

        # 断言实际生成的透视表与预期的透视表相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，验证在没有指定列参数时，数据透视表的行为是否符合预期
    def test_pivot_table_nocols(self):
        # 创建一个包含 'rows', 'cols', 'values' 列的数据框
        df = DataFrame(
            {"rows": ["a", "b", "c"], "cols": ["x", "y", "z"], "values": [1, 2, 3]}
        )

        # 根据 'cols' 列进行聚合求和，生成透视表，列为 'cols'
        rs = df.pivot_table(columns="cols", aggfunc="sum")

        # 根据 'cols' 列进行聚合求和，生成透视表，行为 'cols'，然后转置
        xp = df.pivot_table(index="cols", aggfunc="sum").T

        # 断言两个透视表相等
        tm.assert_frame_equal(rs, xp)

        # 根据 'cols' 列进行聚合计算平均值，生成透视表，列为 'cols'
        rs = df.pivot_table(columns="cols", aggfunc={"values": "mean"})

        # 根据 'cols' 列进行聚合计算平均值，生成透视表，行为 'cols'，然后转置
        xp = df.pivot_table(index="cols", aggfunc={"values": "mean"}).T

        # 断言两个透视表相等
        tm.assert_frame_equal(rs, xp)

    # 定义一个测试函数，验证在不删除缺失值的情况下，数据透视表的行为是否符合预期
    def test_pivot_table_dropna(self):
        # 创建一个包含 'amount', 'customer', 'month', 'product', 'quantity' 列的数据框
        df = DataFrame(
            {
                "amount": {0: 60000, 1: 100000, 2: 50000, 3: 30000},
                "customer": {0: "A", 1: "A", 2: "B", 3: "C"},
                "month": {0: 201307, 1: 201309, 2: 201308, 3: 201310},
                "product": {0: "a", 1: "b", 2: "c", 3: "d"},
                "quantity": {0: 2000000, 1: 500000, 2: 1000000, 3: 1000000},
            }
        )

        # 根据 ('customer', 'product') 和 'month' 列进行聚合求和，生成透视表，列包含 NaN 值
        pv_col = df.pivot_table(
            "quantity", "month", ["customer", "product"], dropna=False
        )

        # 根据 ('customer', 'product') 和 'month' 列进行聚合求和，生成透视表，行包含 NaN 值
        pv_ind = df.pivot_table(
            "quantity", ["customer", "product"], "month", dropna=False
        )

        # 创建一个包含多级索引的预期索引 m
        m = MultiIndex.from_tuples(
            [
                ("A", "a"),
                ("A", "b"),
                ("A", "c"),
                ("A", "d"),
                ("B", "a"),
                ("B", "b"),
                ("B", "c"),
                ("B", "d"),
                ("C", "a"),
                ("C", "b"),
                ("C", "c"),
                ("C", "d"),
            ],
            names=["customer", "product"],
        )

        # 断言 pv_col 的列索引与预期索引 m 相等
        tm.assert_index_equal(pv_col.columns, m)

        # 断言 pv_ind 的行索引与预期索引 m 相等
        tm.assert_index_equal(pv_ind.index, m)
    def test_pivot_table_categorical(self):
        cat1 = Categorical(
            ["a", "a", "b", "b"], categories=["a", "b", "z"], ordered=True
        )
        cat2 = Categorical(
            ["c", "d", "c", "d"], categories=["c", "d", "y"], ordered=True
        )
        # 创建一个包含分类数据的DataFrame，包括列'A'和'B'以及数值列'values'
        df = DataFrame({"A": cat1, "B": cat2, "values": [1, 2, 3, 4]})
        # 调用pivot_table函数生成结果
        result = pivot_table(
            df, values="values", index=["A", "B"], dropna=True, observed=False
        )

        # 生成预期的多级索引
        exp_index = MultiIndex.from_arrays([cat1, cat2], names=["A", "B"])
        # 创建预期的DataFrame，包括'value'列和预期的多级索引
        expected = DataFrame({"values": [1.0, 2.0, 3.0, 4.0]}, index=exp_index)
        # 使用断言函数验证结果和预期是否相等
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_dropna_categoricals(self, dropna):
        # GH 15193
        # 定义分类的类别列表
        categories = ["a", "b", "c", "d"]

        # 创建包含三列'A'、'B'、'C'的DataFrame
        df = DataFrame(
            {
                "A": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
                "B": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "C": range(9),
            }
        )

        # 将'A'列转换为分类类型，并指定类别和排序属性
        df["A"] = df["A"].astype(CategoricalDtype(categories, ordered=False))
        # 使用pivot_table方法生成结果
        result = df.pivot_table(
            index="B", columns="A", values="C", dropna=dropna, observed=False
        )
        # 生成预期的列索引
        expected_columns = Series(["a", "b", "c"], name="A")
        expected_columns = expected_columns.astype(
            CategoricalDtype(categories, ordered=False)
        )
        # 生成预期的行索引
        expected_index = Series([1, 2, 3], name="B")
        # 创建预期的DataFrame，包括预期的行、列索引和数值
        expected = DataFrame(
            [[0.0, 3.0, 6.0], [1.0, 4.0, 7.0], [2.0, 5.0, 8.0]],
            index=expected_index,
            columns=expected_columns,
        )
        # 如果dropna为False，则添加未观察到的分类以进行比较
        if not dropna:
            expected = expected.reindex(columns=Categorical(categories)).astype("float")

        # 使用断言函数验证结果和预期是否相等
        tm.assert_frame_equal(result, expected)

    def test_pivot_with_non_observable_dropna(self, dropna):
        # gh-21133
        # 创建包含两列'A'和'B'的DataFrame，其中'A'列包含分类数据
        df = DataFrame(
            {
                "A": Categorical(
                    [np.nan, "low", "high", "low", "high"],
                    categories=["low", "high"],
                    ordered=True,
                ),
                "B": [0.0, 1.0, 2.0, 3.0, 4.0],
            }
        )

        # 使用pivot_table方法生成结果
        result = df.pivot_table(index="A", values="B", dropna=dropna, observed=False)
        # 如果dropna为True，则生成预期的值和代码
        if dropna:
            values = [2.0, 3.0]
            codes = [0, 1]
        else:
            # 如果dropna为False，添加未观察到的分类以进行比较
            values = [2.0, 3.0, 0.0]
            codes = [0, 1, -1]
        # 创建预期的DataFrame，包括预期的值和分类代码
        expected = DataFrame(
            {"B": values},
            index=Index(
                Categorical.from_codes(codes, categories=["low", "high"], ordered=True),
                name="A",
            ),
        )

        # 使用断言函数验证结果和预期是否相等
        tm.assert_frame_equal(result, expected)
    def test_pivot_with_non_observable_dropna_multi_cat(self, dropna):
        # 测试非可观察类别的数据透视表功能，处理缺失值选项 dropna
        df = DataFrame(
            {
                "A": Categorical(
                    ["left", "low", "high", "low", "high"],
                    categories=["low", "high", "left"],
                    ordered=True,
                ),
                "B": range(5),
            }
        )

        # 使用 pivot_table 方法创建透视表，以列 'A' 为索引，列 'B' 为值，根据 dropna 和 observed 参数确定行为
        result = df.pivot_table(index="A", values="B", dropna=dropna, observed=False)
        # 期望的透视表结果
        expected = DataFrame(
            {"B": [2.0, 3.0, 0.0]},
            index=Index(
                Categorical.from_codes(
                    [0, 1, 2], categories=["low", "high", "left"], ordered=True
                ),
                name="A",
            ),
        )
        if not dropna:
            # 如果 dropna 为 False，将期望结果中的数值列转换为浮点型
            expected["B"] = expected["B"].astype(float)

        # 使用 assert_frame_equal 检查计算结果与期望结果是否一致
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "left_right", [([0] * 4, [1] * 4), (range(3), range(1, 4))]
    )
    def test_pivot_with_interval_index(self, left_right, dropna, closed):
        # GH 25814
        # 根据 GH 25814，测试使用区间索引的数据透视表功能，处理 dropna 和 closed 参数
        left, right = left_right
        interval_values = Categorical(pd.IntervalIndex.from_arrays(left, right, closed))
        df = DataFrame({"A": interval_values, "B": 1})

        # 使用 pivot_table 方法创建透视表，以列 'A' 为索引，列 'B' 为值，根据 dropna 和 observed 参数确定行为
        result = df.pivot_table(index="A", values="B", dropna=dropna, observed=False)
        # 期望的透视表结果
        expected = DataFrame(
            {"B": 1.0}, index=Index(interval_values.unique(), name="A")
        )
        if not dropna:
            # 如果 dropna 为 False，将期望结果转换为浮点型
            expected = expected.astype(float)
        # 使用 assert_frame_equal 检查计算结果与期望结果是否一致
        tm.assert_frame_equal(result, expected)

    def test_pivot_with_interval_index_margins(self):
        # GH 25815
        # 根据 GH 25815，测试使用区间索引的数据透视表功能，包括边际统计
        ordered_cat = pd.IntervalIndex.from_arrays([0, 0, 1, 1], [1, 1, 2, 2])
        df = DataFrame(
            {
                "A": np.arange(4, 0, -1, dtype=np.intp),
                "B": ["a", "b", "a", "b"],
                "C": Categorical(ordered_cat, ordered=True).sort_values(
                    ascending=False
                ),
            }
        )

        # 使用 pivot_table 函数创建透视表，以列 'C' 为索引，列 'B' 为列，列 'A' 为值，包括边际统计
        pivot_tab = pivot_table(
            df,
            index="C",
            columns="B",
            values="A",
            aggfunc="sum",
            margins=True,
            observed=False,
        )

        # 提取透视表中的 "All" 列，作为测试结果
        result = pivot_tab["All"]
        # 期望的 "All" 列结果
        expected = Series(
            [3, 7, 10],
            index=Index([pd.Interval(0, 1), pd.Interval(1, 2), "All"], name="C"),
            name="All",
            dtype=np.intp,
        )
        # 使用 assert_series_equal 检查计算结果与期望结果是否一致
        tm.assert_series_equal(result, expected)

    def test_pass_array(self, data):
        # 测试使用数组进行数据透视表操作
        result = data.pivot_table("D", index=data.A, columns=data.C)
        # 期望的透视表结果
        expected = data.pivot_table("D", index="A", columns="C")
        # 使用 assert_frame_equal 检查计算结果与期望结果是否一致
        tm.assert_frame_equal(result, expected)

    def test_pass_function(self, data):
        # 测试使用函数进行数据透视表操作
        result = data.pivot_table("D", index=lambda x: x // 5, columns=data.C)
        # 期望的透视表结果
        expected = data.pivot_table("D", index=data.index // 5, columns="C")
        # 使用 assert_frame_equal 检查计算结果与期望结果是否一致
        tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，用于测试多维数据的透视表生成
    def test_pivot_table_multiple(self, data):
        # 指定透视表的行索引为 ["A", "B"]，列索引为 "C"
        index = ["A", "B"]
        columns = "C"
        # 调用透视表函数生成透视表
        table = pivot_table(data, index=index, columns=columns)
        # 使用数据的分组平均值来生成期望的透视表
        expected = data.groupby(index + [columns]).agg("mean").unstack()
        # 断言生成的透视表与期望的透视表相等
        tm.assert_frame_equal(table, expected)

    # 定义一个测试函数，用于测试透视表生成后的数据类型转换
    def test_pivot_dtypes(self):
        # 创建一个数据框，包含列 "a"、"v"、"i"，用于测试数据类型转换
        f = DataFrame(
            {
                "a": ["cat", "bat", "cat", "bat"],
                "v": [1, 2, 3, 4],
                "i": ["a", "b", "a", "b"],
            }
        )
        # 断言列 "v" 的数据类型为 int64
        assert f.dtypes["v"] == "int64"

        # 生成透视表，计算 "v" 列在索引为 ["a"]、列为 ["i"] 的情况下的总和
        z = pivot_table(
            f, values="v", index=["a"], columns=["i"], fill_value=0, aggfunc="sum"
        )
        # 获取透视表的列数据类型
        result = z.dtypes
        # 创建一个期望的数据类型序列，期望 "int64" 类型
        expected = Series([np.dtype("int64")] * 2, index=Index(list("ab"), name="i"))
        # 断言生成的透视表的数据类型与期望的数据类型序列相等
        tm.assert_series_equal(result, expected)

        # 创建一个数据框，包含列 "a"、"v"、"i"，用于测试数据类型转换
        f = DataFrame(
            {
                "a": ["cat", "bat", "cat", "bat"],
                "v": [1.5, 2.5, 3.5, 4.5],
                "i": ["a", "b", "a", "b"],
            }
        )
        # 断言列 "v" 的数据类型为 float64
        assert f.dtypes["v"] == "float64"

        # 生成透视表，计算 "v" 列在索引为 ["a"]、列为 ["i"] 的情况下的平均值
        z = pivot_table(
            f, values="v", index=["a"], columns=["i"], fill_value=0, aggfunc="mean"
        )
        # 获取透视表的列数据类型
        result = z.dtypes
        # 创建一个期望的数据类型序列，期望 "float64" 类型
        expected = Series([np.dtype("float64")] * 2, index=Index(list("ab"), name="i"))
        # 断言生成的透视表的数据类型与期望的数据类型序列相等
        tm.assert_series_equal(result, expected)

    # 使用参数化测试标记，定义一个测试函数，用于测试透视表生成后保持列数据类型
    @pytest.mark.parametrize(
        "columns,values",
        [
            ("bool1", ["float1", "float2"]),
            ("bool1", ["float1", "float2", "bool1"]),
            ("bool2", ["float1", "float2", "bool1"]),
        ],
    )
    def test_pivot_preserve_dtypes(self, columns, values):
        # GH 7142 regression test
        # 创建一个数据框 df，包含列 "float1"、"float2"、"bool1"、"bool2"，用于测试透视表生成后保持列数据类型
        v = np.arange(5, dtype=np.float64)
        df = DataFrame(
            {"float1": v, "float2": v + 2.0, "bool1": v <= 2, "bool2": v <= 3}
        )

        # 重置索引后生成透视表，索引为 "index"，列为参数传入的 columns 和 values
        df_res = df.reset_index().pivot_table(
            index="index", columns=columns, values=values
        )

        # 获取透视表各列的数据类型
        result = dict(df_res.dtypes)
        # 创建一个期望的数据类型字典，所有列期望的数据类型为 "float64"
        expected = {col: np.dtype("float64") for col in df_res}
        # 断言生成的透视表的各列数据类型与期望的数据类型字典相等
        assert result == expected
    def test_pivot_no_values(self):
        # GH 14380
        # 创建一个包含日期时间索引的 Pandas DatetimeIndex 对象
        idx = pd.DatetimeIndex(
            ["2011-01-01", "2011-02-01", "2011-01-02", "2011-01-01", "2011-01-02"]
        )
        # 使用 DataFrame 创建一个数据框，指定'A'列的值和日期时间索引
        df = DataFrame({"A": [1, 2, 3, 4, 5]}, index=idx)
        # 对数据框进行数据透视，以月份为索引，以日期为列名
        res = df.pivot_table(index=df.index.month, columns=df.index.day)

        # 期望的列名使用 MultiIndex 进行创建，并设置第二级别为整数类型
        exp_columns = MultiIndex.from_tuples([("A", 1), ("A", 2)])
        exp_columns = exp_columns.set_levels(
            exp_columns.levels[1].astype(np.int32), level=1
        )
        # 创建期望结果的数据框，指定索引和列
        exp = DataFrame(
            [[2.5, 4.0], [2.0, np.nan]],
            index=Index([1, 2], dtype=np.int32),
            columns=exp_columns,
        )
        # 使用测试工具函数验证结果是否符合期望
        tm.assert_frame_equal(res, exp)

        # 使用带有日期时间列的新数据框创建 df
        df = DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "dt": date_range("2011-01-01", freq="D", periods=5),
            },
            index=idx,
        )
        # 对数据框进行数据透视，以月份为索引，使用日期时间列进行分组
        res = df.pivot_table(index=df.index.month, columns=Grouper(key="dt", freq="ME"))
        # 创建期望的列名，使用 MultiIndex 从数组创建，并指定列名为日期时间
        exp_columns = MultiIndex.from_arrays(
            [["A"], pd.DatetimeIndex(["2011-01-31"], dtype="M8[ns]")],
            names=[None, "dt"],
        )
        # 创建期望结果的数据框，指定索引和列
        exp = DataFrame(
            [3.25, 2.0], index=Index([1, 2], dtype=np.int32), columns=exp_columns
        )
        # 使用测试工具函数验证结果是否符合期望
        tm.assert_frame_equal(res, exp)

        # 对数据框进行数据透视，以年度为索引，使用日期时间列进行分组
        res = df.pivot_table(
            index=Grouper(freq="YE"), columns=Grouper(key="dt", freq="ME")
        )
        # 创建期望结果的数据框，指定索引和列
        exp = DataFrame(
            [3.0],
            index=pd.DatetimeIndex(["2011-12-31"], freq="YE"),
            columns=exp_columns,
        )
        # 使用测试工具函数验证结果是否符合期望
        tm.assert_frame_equal(res, exp)
    # 定义一个测试方法，用于测试带有 NaN 值的数据集的数据透视表功能
    def test_pivot_index_with_nan(self, method):
        # GH 3588
        # 定义 NaN 值的常量
        nan = np.nan
        # 创建一个 DataFrame 对象，包含三列数据：a 列包含字符串和 NaN 值，b 列包含字符串，c 列包含整数
        df = DataFrame(
            {
                "a": ["R1", "R2", nan, "R4"],
                "b": ["C1", "C2", "C3", "C4"],
                "c": [10, 15, 17, 20],
            }
        )
        # 根据 method 参数决定使用 DataFrame 的 pivot 方法或者 pd.pivot 函数进行数据透视
        if method:
            result = df.pivot(index="a", columns="b", values="c")
        else:
            result = pd.pivot(df, index="a", columns="b", values="c")
        # 创建期望的 DataFrame 对象，包含 NaN 值，并设定正确的行和列索引
        expected = DataFrame(
            [
                [nan, nan, 17, nan],
                [10, nan, nan, nan],
                [nan, 15, nan, nan],
                [nan, nan, nan, 20],
            ],
            index=Index([nan, "R1", "R2", "R4"], name="a"),
            columns=Index(["C1", "C2", "C3", "C4"], name="b"),
        )
        # 使用测试框架中的方法验证结果与期望是否一致
        tm.assert_frame_equal(result, expected)
        # 使用测试框架中的方法验证另一种数据透视的结果与期望是否一致，并转置期望的结果
        tm.assert_frame_equal(df.pivot(index="b", columns="a", values="c"), expected.T)

    @pytest.mark.parametrize("method", [True, False])
    # 定义另一个测试方法，用于测试带有 NaN 值和日期的数据集的数据透视表功能
    def test_pivot_index_with_nan_dates(self, method):
        # GH9491
        # 创建一个 DataFrame 对象，包含两列数据：a 列包含日期范围，c 列包含整数
        df = DataFrame(
            {
                "a": date_range("2014-02-01", periods=6, freq="D"),
                "c": 100 + np.arange(6),
            }
        )
        # 根据日期计算 b 列，并插入 NaN 值
        df["b"] = df["a"] - pd.Timestamp("2014-02-02")
        df.loc[1, "a"] = df.loc[3, "a"] = np.nan
        df.loc[1, "b"] = df.loc[4, "b"] = np.nan

        # 根据 method 参数决定使用 DataFrame 的 pivot 方法或者 pd.pivot 函数进行数据透视
        if method:
            pv = df.pivot(index="a", columns="b", values="c")
        else:
            pv = pd.pivot(df, index="a", columns="b", values="c")
        # 验证数据透视表中非 NaN 值的数量是否与原始 DataFrame 的长度相同
        assert pv.notna().values.sum() == len(df)

        # 遍历原始 DataFrame 的每一行，验证数据透视表中相应位置的值是否等于原始数据的 c 列的值
        for _, row in df.iterrows():
            assert pv.loc[row["a"], row["b"]] == row["c"]

        # 根据 method 参数决定使用 DataFrame 的 pivot 方法或者 pd.pivot 函数进行另一种数据透视，并转置结果
        if method:
            result = df.pivot(index="b", columns="a", values="c")
        else:
            result = pd.pivot(df, index="b", columns="a", values="c")
        # 使用测试框架中的方法验证结果与第一次数据透视的结果转置后是否一致
        tm.assert_frame_equal(result, pv.T)

    @pytest.mark.parametrize("method", [True, False])
    # 定义一个测试函数，测试带有时区信息的数据透视功能
    def test_pivot_with_tz(self, method, unit):
        # GH 5878：GitHub 上的 issue 编号
        # 创建一个 DataFrame 对象，包含不同时区的日期时间索引和数据列
        df = DataFrame(
            {
                "dt1": pd.DatetimeIndex(
                    [
                        datetime(2013, 1, 1, 9, 0),  # US/Pacific 时区的日期时间
                        datetime(2013, 1, 2, 9, 0),  # US/Pacific 时区的日期时间
                        datetime(2013, 1, 1, 9, 0),  # US/Pacific 时区的日期时间
                        datetime(2013, 1, 2, 9, 0),  # US/Pacific 时区的日期时间
                    ],
                    dtype=f"M8[{unit}, US/Pacific]",  # 使用指定单位和时区的日期时间类型
                ),
                "dt2": pd.DatetimeIndex(
                    [
                        datetime(2014, 1, 1, 9, 0),  # Asia/Tokyo 时区的日期时间
                        datetime(2014, 1, 1, 9, 0),  # Asia/Tokyo 时区的日期时间
                        datetime(2014, 1, 2, 9, 0),  # Asia/Tokyo 时区的日期时间
                        datetime(2014, 1, 2, 9, 0),  # Asia/Tokyo 时区的日期时间
                    ],
                    dtype=f"M8[{unit}, Asia/Tokyo]",  # 使用指定单位和时区的日期时间类型
                ),
                "data1": np.arange(4, dtype="int64"),  # 数据列1
                "data2": np.arange(4, dtype="int64"),  # 数据列2
            }
        )

        # 期望的列索引1，由 data1 和 data2 组成
        exp_col1 = Index(["data1", "data1", "data2", "data2"])
        # 期望的列索引2，包含特定格式的日期时间字符串和时区信息
        exp_col2 = pd.DatetimeIndex(
            ["2014/01/01 09:00", "2014/01/02 09:00"] * 2,
            name="dt2",
            dtype=f"M8[{unit}, Asia/Tokyo]",  # 使用指定单位和时区的日期时间类型
        )
        # 组合成的多级索引
        exp_col = MultiIndex.from_arrays([exp_col1, exp_col2])
        # 期望的行索引，包含特定格式的日期时间字符串和时区信息
        exp_idx = pd.DatetimeIndex(
            ["2013/01/01 09:00", "2013/01/02 09:00"],
            name="dt1",
            dtype=f"M8[{unit}, US/Pacific]",  # 使用指定单位和时区的日期时间类型
        )
        # 创建期望的 DataFrame 对象，包含预期的数据、行索引和列索引
        expected = DataFrame(
            [[0, 2, 0, 2], [1, 3, 1, 3]],
            index=exp_idx,
            columns=exp_col,
        )

        # 根据不同的方法调用数据透视功能，生成数据透视表 pv，并进行结果比较
        if method:
            pv = df.pivot(index="dt1", columns="dt2")  # 使用测试数据进行数据透视
        else:
            pv = pd.pivot(df, index="dt1", columns="dt2")  # 使用 pd.pivot 进行数据透视
        tm.assert_frame_equal(pv, expected)  # 断言生成的数据透视表与预期的表格相等

        # 创建另一个期望的 DataFrame 对象，只包含部分列索引和数据列 data1
        expected = DataFrame(
            [[0, 2], [1, 3]],
            index=exp_idx,
            columns=exp_col2[:2],
        )

        # 根据不同的方法调用数据透视功能，生成数据透视表 pv，并进行结果比较
        if method:
            pv = df.pivot(index="dt1", columns="dt2", values="data1")  # 使用测试数据进行数据透视
        else:
            pv = pd.pivot(df, index="dt1", columns="dt2", values="data1")  # 使用 pd.pivot 进行数据透视
        tm.assert_frame_equal(pv, expected)  # 断言生成的数据透视表与预期的表格相等
    # 定义一个测试方法，用于测试带有时区信息的时间戳在数据框中的处理
    def test_pivot_tz_in_values(self):
        # GH 14948: 关联GitHub问题编号
        # 创建一个包含时间戳和时区信息的数据框
        df = DataFrame(
            [
                {
                    "uid": "aa",
                    "ts": pd.Timestamp("2016-08-12 13:00:00-0700", tz="US/Pacific"),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp("2016-08-12 08:00:00-0700", tz="US/Pacific"),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp("2016-08-12 14:00:00-0700", tz="US/Pacific"),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp("2016-08-25 11:00:00-0700", tz="US/Pacific"),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp("2016-08-25 13:00:00-0700", tz="US/Pacific"),
                },
            ]
        )

        # 将数据框按时间戳列设为索引，然后重置索引
        df = df.set_index("ts").reset_index()
        
        # 创建一个新的时间戳序列，将每个时间戳的时、分、秒和微秒归零
        mins = df.ts.map(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))

        # 对重置索引后的数据框进行数据透视表操作，以时间戳列为值，uid列为索引，mins列为列，使用min函数进行聚合
        result = pivot_table(
            df.set_index("ts").reset_index(),
            values="ts",
            index=["uid"],
            columns=[mins],
            aggfunc="min",
        )

        # 创建预期的结果数据框，包含预期的时间戳和列索引
        expected = DataFrame(
            [
                [
                    pd.Timestamp("2016-08-12 08:00:00-0700", tz="US/Pacific"),
                    pd.Timestamp("2016-08-25 11:00:00-0700", tz="US/Pacific"),
                ]
            ],
            index=Index(["aa"], name="uid"),
            columns=pd.DatetimeIndex(
                [
                    pd.Timestamp("2016-08-12 00:00:00", tz="US/Pacific"),
                    pd.Timestamp("2016-08-25 00:00:00", tz="US/Pacific"),
                ],
                name="ts",
            ),
        )

        # 使用测试框架中的断言方法检查计算结果是否与预期一致
        tm.assert_frame_equal(result, expected)

    # 使用pytest的参数化装饰器定义一个参数化的测试方法
    @pytest.mark.parametrize("method", [True, False])
    # 定义一个测试方法，用于测试周期数据的透视表生成
    def test_pivot_periods(self, method):
        # 创建包含周期数据的数据框
        df = DataFrame(
            {
                "p1": [
                    pd.Period("2013-01-01", "D"),
                    pd.Period("2013-01-02", "D"),
                    pd.Period("2013-01-01", "D"),
                    pd.Period("2013-01-02", "D"),
                ],
                "p2": [
                    pd.Period("2013-01", "M"),
                    pd.Period("2013-01", "M"),
                    pd.Period("2013-02", "M"),
                    pd.Period("2013-02", "M"),
                ],
                "data1": np.arange(4, dtype="int64"),
                "data2": np.arange(4, dtype="int64"),
            }
        )

        # 期望的列索引
        exp_col1 = Index(["data1", "data1", "data2", "data2"])
        exp_col2 = pd.PeriodIndex(["2013-01", "2013-02"] * 2, name="p2", freq="M")
        exp_col = MultiIndex.from_arrays([exp_col1, exp_col2])
        # 期望的数据框
        expected = DataFrame(
            [[0, 2, 0, 2], [1, 3, 1, 3]],
            index=pd.PeriodIndex(["2013-01-01", "2013-01-02"], name="p1", freq="D"),
            columns=exp_col,
        )
        # 根据方法选择不同的透视表生成方式
        if method:
            pv = df.pivot(index="p1", columns="p2")
        else:
            pv = pd.pivot(df, index="p1", columns="p2")
        # 断言生成的透视表与期望的数据框相等
        tm.assert_frame_equal(pv, expected)

        # 另一个期望的数据框
        expected = DataFrame(
            [[0, 2], [1, 3]],
            index=pd.PeriodIndex(["2013-01-01", "2013-01-02"], name="p1", freq="D"),
            columns=pd.PeriodIndex(["2013-01", "2013-02"], name="p2", freq="M"),
        )
        # 根据方法选择不同的透视表生成方式，这次指定数据列为"data1"
        if method:
            pv = df.pivot(index="p1", columns="p2", values="data1")
        else:
            pv = pd.pivot(df, index="p1", columns="p2", values="data1")
        # 断言生成的透视表与期望的数据框相等
        tm.assert_frame_equal(pv, expected)

    # 定义一个测试方法，测试带有合计项的周期数据透视表生成
    def test_pivot_periods_with_margins(self):
        # GH 28323
        # 创建包含周期数据的数据框
        df = DataFrame(
            {
                "a": [1, 1, 2, 2],
                "b": [
                    pd.Period("2019Q1"),
                    pd.Period("2019Q2"),
                    pd.Period("2019Q1"),
                    pd.Period("2019Q2"),
                ],
                "x": 1.0,
            }
        )

        # 期望的数据框
        expected = DataFrame(
            data=1.0,
            index=Index([1, 2, "All"], name="a"),
            columns=Index([pd.Period("2019Q1"), pd.Period("2019Q2"), "All"], name="b"),
        )

        # 生成带有合计项的透视表
        result = df.pivot_table(index="a", columns="b", values="x", margins=True)
        # 断言生成的透视表与期望的数据框相等
        tm.assert_frame_equal(expected, result)

    # 使用参数化测试装饰器，测试不同数据类型和方法对周期数据的透视表生成
    @pytest.mark.parametrize("box", [list, np.array, Series, Index])
    @pytest.mark.parametrize("method", [True, False])
    # 定义一个测试方法，用于测试带有列表类似值的数据透视
    def test_pivot_with_list_like_values(self, box, method):
        # issue #17160: 解决问题 #17160
        # 创建一个列表类似值的对象
        values = box(["baz", "zoo"])
        # 创建一个包含多列数据的数据帧
        df = DataFrame(
            {
                "foo": ["one", "one", "one", "two", "two", "two"],
                "bar": ["A", "B", "C", "A", "B", "C"],
                "baz": [1, 2, 3, 4, 5, 6],
                "zoo": ["x", "y", "z", "q", "w", "t"],
            }
        )

        # 根据方法选择进行数据透视
        if method:
            # 使用 DataFrame 的 pivot 方法进行数据透视
            result = df.pivot(index="foo", columns="bar", values=values)
        else:
            # 使用 pd.pivot 函数进行数据透视
            result = pd.pivot(df, index="foo", columns="bar", values=values)

        # 创建预期的数据
        data = [[1, 2, 3, "x", "y", "z"], [4, 5, 6, "q", "w", "t"]]
        # 创建预期的索引
        index = Index(data=["one", "two"], name="foo")
        # 创建预期的多级列索引
        columns = MultiIndex(
            levels=[["baz", "zoo"], ["A", "B", "C"]],
            codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
            names=[None, "bar"],
        )
        # 创建预期的数据帧
        expected = DataFrame(data=data, index=index, columns=columns)
        # 将预期结果中的 'baz' 列转换为对象类型
        expected["baz"] = expected["baz"].astype(object)
        # 使用测试模块中的 assert_frame_equal 方法比较结果和预期
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "values",
        [
            ["bar", "baz"],
            np.array(["bar", "baz"]),
            Series(["bar", "baz"]),
            Index(["bar", "baz"]),
        ],
    )
    @pytest.mark.parametrize("method", [True, False])
    # 定义一个参数化测试方法，用于测试带有 NaN 值的列表类似值的数据透视
    def test_pivot_with_list_like_values_nans(self, values, method):
        # issue #17160: 解决问题 #17160
        # 创建一个包含多列数据的数据帧
        df = DataFrame(
            {
                "foo": ["one", "one", "one", "two", "two", "two"],
                "bar": ["A", "B", "C", "A", "B", "C"],
                "baz": [1, 2, 3, 4, 5, 6],
                "zoo": ["x", "y", "z", "q", "w", "t"],
            }
        )

        # 根据方法选择进行数据透视
        if method:
            # 使用 DataFrame 的 pivot 方法进行数据透视
            result = df.pivot(index="zoo", columns="foo", values=values)
        else:
            # 使用 pd.pivot 函数进行数据透视
            result = pd.pivot(df, index="zoo", columns="foo", values=values)

        # 创建预期的数据
        data = [
            [np.nan, "A", np.nan, 4],
            [np.nan, "C", np.nan, 6],
            [np.nan, "B", np.nan, 5],
            ["A", np.nan, 1, np.nan],
            ["B", np.nan, 2, np.nan],
            ["C", np.nan, 3, np.nan],
        ]
        # 创建预期的索引
        index = Index(data=["q", "t", "w", "x", "y", "z"], name="zoo")
        # 创建预期的多级列索引
        columns = MultiIndex(
            levels=[["bar", "baz"], ["one", "two"]],
            codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
            names=[None, "foo"],
        )
        # 创建预期的数据帧
        expected = DataFrame(data=data, index=index, columns=columns)
        # 将预期结果中的 'baz' 列转换为对象类型
        expected["baz"] = expected["baz"].astype(object)
        # 使用测试模块中的 assert_frame_equal 方法比较结果和预期
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试当 columns 参数为 None 时是否会抛出错误
    def test_pivot_columns_none_raise_error(self):
        # GH 30924: GitHub 问题 #30924
        # 创建一个包含多列数据的数据帧
        df = DataFrame({"col1": ["a", "b", "c"], "col2": [1, 2, 3], "col3": [1, 2, 3]})
        # 定义预期会抛出的错误消息
        msg = r"pivot\(\) missing 1 required keyword-only argument: 'columns'"
        # 使用 pytest 的断言捕获机制来验证是否会抛出特定类型和消息的错误
        with pytest.raises(TypeError, match=msg):
            df.pivot(index="col1", values="col3")
    @pytest.mark.xfail(
        reason="MultiIndexed unstack with tuple names fails with KeyError GH#19966"
    )
    @pytest.mark.parametrize("method", [True, False])
    # 定义一个测试函数，用于测试带有多级索引的数据透视表功能
    def test_pivot_with_multiindex(self, method):
        # issue #17160
        # 创建一个索引对象，包含整数数据作为索引
        index = Index(data=[0, 1, 2, 3, 4, 5])
        # 定义数据列表，每个子列表代表一行数据，包含字符串、整数和字母
        data = [
            ["one", "A", 1, "x"],
            ["one", "B", 2, "y"],
            ["one", "C", 3, "z"],
            ["two", "A", 4, "q"],
            ["two", "B", 5, "w"],
            ["two", "C", 6, "t"],
        ]
        # 创建多级列索引对象，包含两个级别的标签和代码，表示列的结构
        columns = MultiIndex(
            levels=[["bar", "baz"], ["first", "second"]],
            codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
        )
        # 创建数据框对象，使用给定的数据、索引和列结构
        df = DataFrame(data=data, index=index, columns=columns, dtype="object")
        # 根据方法标志选择使用测试的数据透视表方法
        if method:
            # 使用数据框对象的 pivot 方法生成数据透视表结果
            result = df.pivot(
                index=("bar", "first"),
                columns=("bar", "second"),
                values=("baz", "first"),
            )
        else:
            # 使用 pandas 库的 pivot 函数生成数据透视表结果
            result = pd.pivot(
                df,
                index=("bar", "first"),
                columns=("bar", "second"),
                values=("baz", "first"),
            )

        # 定义预期的数据字典，包含 Series 对象作为数据框的预期输出
        data = {
            "A": Series([1, 4], index=["one", "two"]),
            "B": Series([2, 5], index=["one", "two"]),
            "C": Series([3, 6], index=["one", "two"]),
        }
        # 创建预期的数据框对象，用于与测试结果比较
        expected = DataFrame(data)
        # 使用测试工具类的 assert_frame_equal 方法检查测试结果和预期结果是否一致
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("method", [True, False])
    # 定义一个测试函数，用于测试带有值元组的数据透视表功能
    def test_pivot_with_tuple_of_values(self, method):
        # issue #17160
        # 创建包含字符串和整数数据的数据框对象
        df = DataFrame(
            {
                "foo": ["one", "one", "one", "two", "two", "two"],
                "bar": ["A", "B", "C", "A", "B", "C"],
                "baz": [1, 2, 3, 4, 5, 6],
                "zoo": ["x", "y", "z", "q", "w", "t"],
            }
        )
        # 使用 pytest 的 raises 方法检查是否会抛出 KeyError 异常
        with pytest.raises(KeyError, match=r"^\('bar', 'baz'\)$"):
            # 当元组被视为单个列名时，根据方法标志选择使用测试的数据透视表方法
            if method:
                # 使用数据框对象的 pivot 方法生成数据透视表结果
                df.pivot(index="zoo", columns="foo", values=("bar", "baz"))
            else:
                # 使用 pandas 库的 pivot 函数生成数据透视表结果
                pd.pivot(df, index="zoo", columns="foo", values=("bar", "baz"))

    def _check_output(
        self,
        result,
        values_col,
        data,
        index=None,
        columns=None,
        margins_col="All",
    ):
        # 如果未指定索引列，则默认为 ["A", "B"]
        if index is None:
            index = ["A", "B"]
        # 如果未指定列名，则默认为 ["C"]
        if columns is None:
            columns = ["C"]
        # 获取结果中的列边距
        col_margins = result.loc[result.index[:-1], margins_col]
        # 根据索引分组计算期望的列边距
        expected_col_margins = data.groupby(index)[values_col].mean()
        # 断言结果的列边距与期望的列边距相等，忽略名称检查
        tm.assert_series_equal(col_margins, expected_col_margins, check_names=False)
        # 断言列边距的名称与指定的边距列名相等
        assert col_margins.name == margins_col

        # 对结果进行按索引排序
        result = result.sort_index()
        # 获取索引边距
        index_margins = result.loc[(margins_col, "")].iloc[:-1]

        # 根据列分组计算期望的索引边距
        expected_ix_margins = data.groupby(columns)[values_col].mean()
        # 断言结果的索引边距与期望的索引边距相等，忽略名称检查
        tm.assert_series_equal(index_margins, expected_ix_margins, check_names=False)
        # 断言索引边距的名称与指定的边距列名为 (margins_col, "")
        assert index_margins.name == (margins_col, "")

        # 获取总计边距
        grand_total_margins = result.loc[(margins_col, ""), margins_col]
        # 计算期望的总计边距
        expected_total_margins = data[values_col].mean()
        # 断言总计边距与期望的总计边距相等
        assert grand_total_margins == expected_total_margins

    def test_margins(self, data):
        # 指定了列名
        result = data.pivot_table(
            values="D", index=["A", "B"], columns="C", margins=True, aggfunc="mean"
        )
        # 检查输出结果
        self._check_output(result, "D", data)

        # 设置不同的边距名称（非 'All'）
        result = data.pivot_table(
            values="D",
            index=["A", "B"],
            columns="C",
            margins=True,
            aggfunc="mean",
            margins_name="Totals",
        )
        # 检查输出结果，使用自定义的边距列名
        self._check_output(result, "D", data, margins_col="Totals")

        # 未指定列名
        table = data.pivot_table(
            index=["A", "B"], columns="C", margins=True, aggfunc="mean"
        )
        # 遍历表格中的每个值列
        for value_col in table.columns.levels[0]:
            # 检查输出结果
            self._check_output(table[value_col], value_col, data)

    def test_no_col(self, data):
        # 未指定列

        # 为了解决一个小bug
        data.columns = [k * 2 for k in data.columns]
        # 匹配错误消息以帮助调试
        msg = re.escape("agg function failed [how->mean,dtype->")
        # 断言引发 TypeError 异常，消息匹配特定模式
        with pytest.raises(TypeError, match=msg):
            data.pivot_table(index=["AA", "BB"], margins=True, aggfunc="mean")
        # 计算表格并删除 "CC" 列，生成表格
        table = data.drop(columns="CC").pivot_table(
            index=["AA", "BB"], margins=True, aggfunc="mean"
        )
        # 遍历表格中的每个值列
        for value_col in table.columns:
            # 获取总计，检查总计是否等于数据列的平均值
            totals = table.loc[("All", ""), value_col]
            assert totals == data[value_col].mean()

        # 再次引发 TypeError 异常，消息匹配特定模式
        with pytest.raises(TypeError, match=msg):
            data.pivot_table(index=["AA", "BB"], margins=True, aggfunc="mean")
        # 计算表格并删除 "CC" 列，生成表格
        table = data.drop(columns="CC").pivot_table(
            index=["AA", "BB"], margins=True, aggfunc="mean"
        )
        # 遍历每个项目，检查总计是否等于数据列的平均值
        for item in ["DD", "EE", "FF"]:
            totals = table.loc[("All", ""), item]
            assert totals == data[item].mean()
    @pytest.mark.parametrize(
        "columns, aggfunc, values, expected_columns",
        [  # 参数化测试，定义多组参数：columns, aggfunc, values, expected_columns
            (
                "A",  # 第一组参数：列为"A"
                "mean",  # 聚合函数为"mean"
                [[5.5, 5.5, 2.2, 2.2], [8.0, 8.0, 4.4, 4.4]],  # 预期值数据
                Index(["bar", "All", "foo", "All"], name="A"),  # 预期的列名索引
            ),
            (
                ["A", "B"],  # 第二组参数：列为["A", "B"]
                "sum",  # 聚合函数为"sum"
                [
                    [9, 13, 22, 5, 6, 11],  # 第一组预期值数据
                    [14, 18, 32, 11, 11, 22],  # 第二组预期值数据
                ],
                MultiIndex.from_tuples(
                    [  # 预期的多级索引，由元组组成
                        ("bar", "one"),
                        ("bar", "two"),
                        ("bar", "All"),
                        ("foo", "one"),
                        ("foo", "two"),
                        ("foo", "All"),
                    ],
                    names=["A", "B"],  # 索引名称为"A", "B"
                ),
            ),
        ],
    )
    def test_margin_with_only_columns_defined(
        self, columns, aggfunc, values, expected_columns
    ):
        # GH 31016
        df = DataFrame(  # 创建数据框
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": [  # 列"C"的数据
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
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],  # 列"D"的数据
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],  # 列"E"的数据
            }
        )
        if aggfunc != "sum":  # 如果聚合函数不是"sum"
            msg = re.escape("agg function failed [how->mean,dtype->")  # 出错消息
            with pytest.raises(TypeError, match=msg):  # 断言抛出TypeError异常，消息匹配预期
                df.pivot_table(columns=columns, margins=True, aggfunc=aggfunc)  # 调用pivot_table进行测试
        if "B" not in columns:  # 如果"B"不在列名中
            df = df.drop(columns="B")  # 删除数据框中的列"B"
        result = df.drop(columns="C").pivot_table(  # 调用pivot_table生成结果
            columns=columns, margins=True, aggfunc=aggfunc
        )
        expected = DataFrame(values, index=Index(["D", "E"]), columns=expected_columns)  # 生成预期结果数据框

        tm.assert_frame_equal(result, expected)  # 断言结果与预期相等

    def test_margins_dtype(self, data):
        # GH 17013
        df = data.copy()  # 复制数据
        df[["D", "E", "F"]] = np.arange(len(df) * 3).reshape(len(df), 3).astype("i8")  # 修改列"D", "E", "F"的数据类型为'i8'

        mi_val = list(product(["bar", "foo"], ["one", "two"])) + [("All", "")]  # 多级索引的值列表
        mi = MultiIndex.from_tuples(mi_val, names=("A", "B"))  # 创建多级索引
        expected = DataFrame(  # 生成预期的数据框
            {"dull": [12, 21, 3, 9, 45], "shiny": [33, 0, 36, 51, 120]}, index=mi
        ).rename_axis("C", axis=1)
        expected["All"] = expected["dull"] + expected["shiny"]  # 添加"All"列，计算"dull"和"shiny"列的和

        result = df.pivot_table(  # 调用pivot_table生成结果
            values="D",  # 聚合值为"D"
            index=["A", "B"],  # 索引为["A", "B"]
            columns="C",  # 列为"C"
            margins=True,  # 包括边际
            aggfunc="sum",  # 聚合函数为"sum"
            fill_value=0,  # 填充缺失值为0
        )

        tm.assert_frame_equal(expected, result)  # 断言结果与预期相等
    def test_margins_dtype_len(self, data):
        # 创建多级索引的值列表，包括所有可能的组合以及一个特殊的("All", "")组合
        mi_val = list(product(["bar", "foo"], ["one", "two"])) + [("All", "")]
        # 用值列表创建一个多级索引对象，命名为("A", "B")
        mi = MultiIndex.from_tuples(mi_val, names=("A", "B"))
        # 创建预期的数据帧，包括"dull"和"shiny"两列，并设置索引为mi，同时重命名列轴为"C"
        expected = DataFrame(
            {"dull": [1, 1, 2, 1, 5], "shiny": [2, 0, 2, 2, 6]}, index=mi
        ).rename_axis("C", axis=1)
        # 添加一个"All"列，其值为"dull"和"shiny"列对应行的和
        expected["All"] = expected["dull"] + expected["shiny"]

        # 对给定数据进行数据透视表操作，以"D"列为值，("A", "B")为索引，"C"为列，并计算长度，包括边际
        result = data.pivot_table(
            values="D",
            index=["A", "B"],
            columns="C",
            margins=True,
            aggfunc=len,
            fill_value=0,
        )

        # 使用测试工具验证预期输出与实际结果是否相等
        tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize("cols", [(1, 2), ("a", "b"), (1, "b"), ("a", 1)])
    def test_pivot_table_multiindex_only(self, cols):
        # GitHub issue 17038的测试情况
        # 创建包含指定列的数据帧，列名为cols[0]和cols[1]，还有一个"v"列
        df2 = DataFrame({cols[0]: [1, 2, 3], cols[1]: [1, 2, 3], "v": [4, 5, 6]})

        # 对数据帧进行数据透视表操作，以"v"列为值，使用cols作为多级索引的列
        result = df2.pivot_table(values="v", columns=cols)
        # 创建预期的数据帧，使用cols作为列的多级索引，包含一个行"v"
        expected = DataFrame(
            [[4.0, 5.0, 6.0]],
            columns=MultiIndex.from_tuples([(1, 1), (2, 2), (3, 3)], names=cols),
            index=Index(["v"], dtype=object),
        )

        # 使用测试工具验证预期输出与实际结果是否相等
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_retains_tz(self):
        # 创建带有时区信息的日期时间索引
        dti = date_range("2016-01-01", periods=3, tz="Europe/Amsterdam")
        # 创建数据帧，包含"A"、"B"、"C"列，其中"C"列使用上述日期时间索引
        df = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(3),
                "B": np.random.default_rng(2).standard_normal(3),
                "C": dti,
            }
        )
        # 对数据帧进行数据透视表操作，以["B", "C"]作为索引，保留缺失值
        result = df.pivot_table(index=["B", "C"], dropna=False)

        # 验证结果中的时间索引是否与原始时间索引相同
        assert result.index.levels[1].equals(dti)

    def test_pivot_integer_columns(self):
        # 由于unstack中的上游错误导致的问题

        # 创建包含日期范围和值的数据列表
        d = date.min
        data = list(
            product(
                ["foo", "bar"],
                ["A", "B", "C"],
                ["x1", "x2"],
                [d + timedelta(i) for i in range(20)],
                [1.0],
            )
        )
        # 创建数据帧
        df = DataFrame(data)
        # 对数据帧进行数据透视表操作，以第4列的值为值，[0, 1, 3]为索引，[2]为列
        table = df.pivot_table(values=4, index=[0, 1, 3], columns=[2])

        # 对数据帧进行列名转换为字符串类型的操作
        df2 = df.rename(columns=str)
        # 对转换后的数据帧进行数据透视表操作，以"4"列的值为值，["0", "1", "3"]为索引，["2"]为列
        table2 = df2.pivot_table(values="4", index=["0", "1", "3"], columns=["2"])

        # 使用测试工具验证预期输出与实际结果是否相等，不检查列名
        tm.assert_frame_equal(table, table2, check_names=False)

    def test_pivot_no_level_overlap(self):
        # GitHub issue #1181的测试情况

        # 创建包含"a", "b", "c", "value"列的数据帧
        data = DataFrame(
            {
                "a": ["a", "a", "a", "a", "b", "b", "b", "b"] * 2,
                "b": [0, 0, 0, 0, 1, 1, 1, 1] * 2,
                "c": (["foo"] * 4 + ["bar"] * 4) * 2,
                "value": np.random.default_rng(2).standard_normal(16),
            }
        )

        # 对数据帧进行数据透视表操作，以"value"列的值为值，"a"为索引，["b", "c"]为列
        table = data.pivot_table("value", index="a", columns=["b", "c"])

        # 对数据帧进行分组操作，按"a", "b", "c"列分组，并计算"value"列的均值
        grouped = data.groupby(["a", "b", "c"])["value"].mean()
        # 创建预期的数据帧，使用"b"和"c"列的值作为多级索引的列，去除所有值均为空的列
        expected = grouped.unstack("b").unstack("c").dropna(axis=1, how="all")

        # 使用测试工具验证预期输出与实际结果是否相等
        tm.assert_frame_equal(table, expected)
    # 测试函数：测试按词典序排序列的数据透视表生成
    def test_pivot_columns_lexsorted(self):
        # 设置样本数量
        n = 10000
        
        # 定义数据类型，包括索引、符号、年、月、日、数量、价格
        dtype = np.dtype(
            [
                ("Index", object),
                ("Symbol", object),
                ("Year", int),
                ("Month", int),
                ("Day", int),
                ("Quantity", int),
                ("Price", float),
            ]
        )
        
        # 定义产品数组，包括指数和符号，使用指定的数据类型
        products = np.array(
            [
                ("SP500", "ADBE"),
                ("SP500", "NVDA"),
                ("SP500", "ORCL"),
                ("NDQ100", "AAPL"),
                ("NDQ100", "MSFT"),
                ("NDQ100", "GOOG"),
                ("FTSE", "DGE.L"),
                ("FTSE", "TSCO.L"),
                ("FTSE", "GSK.L"),
            ],
            dtype=[("Index", object), ("Symbol", object)],
        )
        
        # 创建一个空的数据数组，使用指定的数据类型
        items = np.empty(n, dtype=dtype)
        
        # 随机生成整数数组，用于选择产品数组的索引
        iproduct = np.random.default_rng(2).integers(0, len(products), n)
        
        # 将随机选择的产品索引映射到数据数组的索引列
        items["Index"] = products["Index"][iproduct]
        items["Symbol"] = products["Symbol"][iproduct]
        
        # 生成日期范围对象，随机选择日期并填充数据数组的年、月、日列
        dr = date_range(date(2000, 1, 1), date(2010, 12, 31))
        dates = dr[np.random.default_rng(2).integers(0, len(dr), n)]
        items["Year"] = dates.year
        items["Month"] = dates.month
        items["Day"] = dates.day
        
        # 使用对数正态分布生成价格数据，填充数据数组的价格列
        items["Price"] = np.random.default_rng(2).lognormal(4.0, 2.0, n)
        
        # 创建数据框对象，基于填充的数据数组
        df = DataFrame(items)
        
        # 生成数据透视表，按价格平均值对月份、日期、指数和符号进行分组
        pivoted = df.pivot_table(
            "Price",
            index=["Month", "Day"],
            columns=["Index", "Symbol", "Year"],
            aggfunc="mean",
        )
        
        # 断言检查生成的透视表列是否单调递增
        assert pivoted.columns.is_monotonic_increasing

    # 测试函数：测试复杂聚合函数的数据透视表生成
    def test_pivot_complex_aggfunc(self, data):
        # 定义聚合函数字典，指定 D 列使用标准差，E 列使用总和
        f = {"D": ["std"], "E": ["sum"]}
        
        # 使用数据的 A、B 列进行分组，并使用指定的聚合函数生成期望的数据框
        expected = data.groupby(["A", "B"]).agg(f).unstack("B")
        
        # 使用数据的 A、B 列生成数据透视表，指定聚合函数为 f
        result = data.pivot_table(index="A", columns="B", aggfunc=f)
        
        # 断言检查生成的数据透视表与期望的数据框是否相等
        tm.assert_frame_equal(result, expected)

    # 测试函数：测试数据透视表生成，不传递值和列
    def test_margins_no_values_no_cols(self, data):
        # 回归测试数据透视表：不传递值或列
        result = data[["A", "B"]].pivot_table(
            index=["A", "B"], aggfunc=len, margins=True
        )
        
        # 将结果转换为列表
        result_list = result.tolist()
        
        # 断言检查结果列表前面几项的总和是否等于最后一项
        assert sum(result_list[:-1]) == result_list[-1]

    # 测试函数：测试数据透视表生成，不传递值但有两行
    def test_margins_no_values_two_rows(self, data):
        # 回归测试数据透视表：不传递值，但行是多级索引
        result = data[["A", "B", "C"]].pivot_table(
            index=["A", "B"], columns="C", aggfunc=len, margins=True
        )
        
        # 断言检查结果的 All 列转换为列表是否等于预期列表
        assert result.All.tolist() == [3.0, 1.0, 4.0, 3.0, 11.0]

    # 测试函数：测试数据透视表生成，不传递值但有一个行和一列
    def test_margins_no_values_one_row_one_col(self, data):
        # 回归测试数据透视表：不传递值，但定义了行和列
        result = data[["A", "B"]].pivot_table(
            index="A", columns="B", aggfunc=len, margins=True
        )
        
        # 断言检查结果的 All 列转换为列表是否等于预期列表
        assert result.All.tolist() == [4.0, 7.0, 11.0]
    def test_margins_no_values_two_row_two_cols(self, data):
        # Regression test on pivot table: no values passed but rows and cols
        # are multi-indexed
        
        # 在数据中添加一列 "D"，内容为 ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
        data["D"] = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
        
        # 生成数据的透视表，使用 ["A", "B"] 作为行索引，["C", "D"]作为列索引，聚合函数为 len，同时计算边际值
        result = data[["A", "B", "C", "D"]].pivot_table(
            index=["A", "B"], columns=["C", "D"], aggfunc=len, margins=True
        )
        
        # 断言透视表结果的 "All" 列的值列表是否等于 [3.0, 1.0, 4.0, 3.0, 11.0]
        assert result.All.tolist() == [3.0, 1.0, 4.0, 3.0, 11.0]

    @pytest.mark.parametrize("margin_name", ["foo", "one", 666, None, ["a", "b"]])
    def test_pivot_table_with_margins_set_margin_name(self, margin_name, data):
        # see gh-3335
        # 生成错误消息，表明参数 margins_name 必须为字符串
        msg = (
            f'Conflicting name "{margin_name}" in margins|'
            "margins_name argument must be a string"
        )
        
        # 使用 pytest 的上下文管理器来检测是否会引发 ValueError，匹配错误消息 msg
        with pytest.raises(ValueError, match=msg):
            # 创建透视表，使用 data 数据，值为 "D"，行索引为 ["A", "B"]，列索引为 ["C"]，计算边际值，并设定边际值的名称为 margin_name
            pivot_table(
                data,
                values="D",
                index=["A", "B"],
                columns=["C"],
                margins=True,
                margins_name=margin_name,
            )
        
        with pytest.raises(ValueError, match=msg):
            # 创建透视表，使用 data 数据，值为 "D"，行索引为 ["C"]，列索引为 ["A", "B"]，计算边际值，并设定边际值的名称为 margin_name
            pivot_table(
                data,
                values="D",
                index=["C"],
                columns=["A", "B"],
                margins=True,
                margins_name=margin_name,
            )
        
        with pytest.raises(ValueError, match=msg):
            # 创建透视表，使用 data 数据，值为 "D"，行索引为 ["A"]，列索引为 ["B"]，计算边际值，并设定边际值的名称为 margin_name
            pivot_table(
                data,
                values="D",
                index=["A"],
                columns=["B"],
                margins=True,
                margins_name=margin_name,
            )
    # 定义一个测试方法，用于测试带有时区的日期时间的数据透视表功能
    def test_pivot_datetime_tz(self):
        # 创建第一个日期时间索引，使用美国太平洋时区，命名为dt1
        dates1 = pd.DatetimeIndex(
            [
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
            ],
            dtype="M8[ns, US/Pacific]",
            name="dt1",
        )
        # 创建第二个日期时间索引，使用亚洲东京时区
        dates2 = pd.DatetimeIndex(
            [
                "2013-01-01 15:00:00",
                "2013-01-01 15:00:00",
                "2013-01-01 15:00:00",
                "2013-02-01 15:00:00",
                "2013-02-01 15:00:00",
                "2013-02-01 15:00:00",
            ],
            dtype="M8[ns, Asia/Tokyo]",
        )
        # 创建包含多个列的数据框，包括'label'、'dt1'、'dt2'、'value1'和'value2'
        df = DataFrame(
            {
                "label": ["a", "a", "a", "b", "b", "b"],
                "dt1": dates1,
                "dt2": dates2,
                "value1": np.arange(6, dtype="int64"),
                "value2": [1, 2] * 3,
            }
        )

        # 期望的第一个索引，使用前三个dt1时间戳
        exp_idx = dates1[:3]
        # 期望的第一个列索引，包含'value1'的标签
        exp_col1 = Index(["value1", "value1"])
        # 期望的第二个列索引，包含'label'的标签
        exp_col2 = Index(["a", "b"], name="label")
        # 期望的列多索引，由exp_col1和exp_col2组成
        exp_col = MultiIndex.from_arrays([exp_col1, exp_col2])
        # 创建第一个期望的数据框，包含值为0到5的'value1'列，使用exp_idx和exp_col作为索引和列
        expected = DataFrame(
            [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]], index=exp_idx, columns=exp_col
        )
        # 执行数据透视表操作，将结果存储在result中
        result = pivot_table(df, index=["dt1"], columns=["label"], values=["value1"])
        # 使用测试模块断言结果与期望值相等
        tm.assert_frame_equal(result, expected)

        # 期望的第一个列索引，包含'sum'和'mean'标签
        exp_col1 = Index(["sum", "sum", "sum", "sum", "mean", "mean", "mean", "mean"])
        # 期望的第二个列索引，包含'value1'和'value2'的标签
        exp_col2 = Index(["value1", "value1", "value2", "value2"] * 2)
        # 期望的第三个列索引，包含与dates2索引相同的日期时间戳，使用亚洲东京时区，命名为dt2
        exp_col3 = pd.DatetimeIndex(
            ["2013-01-01 15:00:00", "2013-02-01 15:00:00"] * 4,
            dtype="M8[ns, Asia/Tokyo]",
            name="dt2",
        )
        # 期望的列多索引，由exp_col1、exp_col2和exp_col3组成
        exp_col = MultiIndex.from_arrays([exp_col1, exp_col2, exp_col3])
        # 创建第一个期望的数据框，包含与value1和value2相关的整数值，使用exp_idx和exp_col的前四列作为索引和列
        expected1 = DataFrame(
            np.array(
                [
                    [
                        0,
                        3,
                        1,
                        2,
                    ],
                    [1, 4, 2, 1],
                    [2, 5, 1, 2],
                ],
                dtype="int64",
            ),
            index=exp_idx,
            columns=exp_col[:4],
        )
        # 创建第二个期望的数据框，包含与value1和value2相关的浮点数值，使用exp_idx和exp_col的后四列作为索引和列
        expected2 = DataFrame(
            np.array(
                [
                    [0.0, 3.0, 1.0, 2.0],
                    [1.0, 4.0, 2.0, 1.0],
                    [2.0, 5.0, 1.0, 2.0],
                ],
            ),
            index=exp_idx,
            columns=exp_col[4:],
        )
        # 合并两个期望的数据框，沿着列方向连接它们，形成完整的期望结果
        expected = concat([expected1, expected2], axis=1)

        # 执行数据透视表操作，将结果存储在result中，使用aggfunc参数指定'sum'和'mean'作为聚合函数
        result = pivot_table(
            df,
            index=["dt1"],
            columns=["dt2"],
            values=["value1", "value2"],
            aggfunc=["sum", "mean"],
        )
        # 使用测试模块断言结果与期望值相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，用于测试 DataFrame 的 pivot_table 方法的不同用例
    def test_pivot_dtaccessor(self):
        # GH 8103 GitHub issue 标识符，可能与问题相关联
        dates1 = pd.DatetimeIndex(
            [
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
            ]
        )
        dates2 = pd.DatetimeIndex(
            [
                "2013-01-01 15:00:00",
                "2013-01-01 15:00:00",
                "2013-01-01 15:00:00",
                "2013-02-01 15:00:00",
                "2013-02-01 15:00:00",
                "2013-02-01 15:00:00",
            ]
        )
        # 创建 DataFrame 对象 df，包括 label, dt1, dt2, value1, value2 列
        df = DataFrame(
            {
                "label": ["a", "a", "a", "b", "b", "b"],
                "dt1": dates1,
                "dt2": dates2,
                "value1": np.arange(6, dtype="int64"),
                "value2": [1, 2] * 3,
            }
        )

        # 调用 pivot_table 方法，对 df 进行数据透视，按 label 和 dt1 的小时部分作为行索引和列索引，value1 作为值
        result = pivot_table(
            df, index="label", columns=df["dt1"].dt.hour, values="value1"
        )

        # 创建预期的索引对象 exp_idx 和 DataFrame 对象 expected，用于验证结果是否符合预期
        exp_idx = Index(["a", "b"], name="label")
        expected = DataFrame(
            {7: [0.0, 3.0], 8: [1.0, 4.0], 9: [2.0, 5.0]},
            index=exp_idx,
            columns=Index([7, 8, 9], dtype=np.int32, name="dt1"),
        )
        # 使用测试工具进行结果验证，确保返回结果与预期的 DataFrame 一致
        tm.assert_frame_equal(result, expected)

        # 再次调用 pivot_table 方法，这次以 dt2 的月份为行索引，dt1 的小时部分为列索引，value1 作为值
        result = pivot_table(
            df, index=df["dt2"].dt.month, columns=df["dt1"].dt.hour, values="value1"
        )

        # 创建预期的 DataFrame 对象 expected，用于验证结果是否符合预期
        expected = DataFrame(
            {7: [0.0, 3.0], 8: [1.0, 4.0], 9: [2.0, 5.0]},
            index=Index([1, 2], dtype=np.int32, name="dt2"),
            columns=Index([7, 8, 9], dtype=np.int32, name="dt1"),
        )
        # 使用测试工具进行结果验证，确保返回结果与预期的 DataFrame 一致
        tm.assert_frame_equal(result, expected)

        # 再次调用 pivot_table 方法，这次以 dt2 的年份为行索引，同时使用 dt1 的小时和 dt2 的月份作为多级列索引，value1 作为值
        result = pivot_table(
            df,
            index=df["dt2"].dt.year.values,
            columns=[df["dt1"].dt.hour, df["dt2"].dt.month],
            values="value1",
        )

        # 创建预期的多级列索引对象 exp_col 和 DataFrame 对象 expected，用于验证结果是否符合预期
        exp_col = MultiIndex.from_arrays(
            [
                np.array([7, 7, 8, 8, 9, 9], dtype=np.int32),
                np.array([1, 2] * 3, dtype=np.int32),
            ],
            names=["dt1", "dt2"],
        )
        expected = DataFrame(
            np.array([[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]]),
            index=Index([2013], dtype=np.int32),
            columns=exp_col,
        )
        # 使用测试工具进行结果验证，确保返回结果与预期的 DataFrame 一致
        tm.assert_frame_equal(result, expected)

        # 最后一次调用 pivot_table 方法，以指定的索引数组和 dt1 的小时和 dt2 的月份作为多级列索引，value1 作为值
        result = pivot_table(
            df,
            index=np.array(["X", "X", "X", "X", "Y", "Y"]),
            columns=[df["dt1"].dt.hour, df["dt2"].dt.month],
            values="value1",
        )
        # 创建预期的 DataFrame 对象 expected，用于验证结果是否符合预期
        expected = DataFrame(
            np.array(
                [[0, 3, 1, np.nan, 2, np.nan], [np.nan, np.nan, np.nan, 4, np.nan, 5]]
            ),
            index=["X", "Y"],
            columns=exp_col,
        )
        # 使用测试工具进行结果验证，确保返回结果与预期的 DataFrame 一致
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于测试按日进行数据透视表操作
    def test_daily(self):
        # 生成一个从 "1/1/2000" 到 "12/31/2004" 的日期范围，频率为每日
        rng = date_range("1/1/2000", "12/31/2004", freq="D")
        # 创建一个时间序列，其值为从 0 开始的递增序列，索引为 rng
        ts = Series(np.arange(len(rng)), index=rng)

        # 对时间序列 ts 执行数据透视操作，按年作为索引，按年中的第几天作为列
        result = pivot_table(
            DataFrame(ts), index=ts.index.year, columns=ts.index.dayofyear
        )
        # 去除结果的列索引的第一层级
        result.columns = result.columns.droplevel(0)

        # 创建一个包含 ts 索引中每天的年中第几天的数组
        doy = np.asarray(ts.index.dayofyear)

        # 期望的结果初始化为空字典
        expected = {}
        # 对时间序列 ts 中每个年份进行循环
        for y in ts.index.year.unique().values:
            # 创建一个布尔掩码，表示 ts 中索引为当前年份 y 的位置
            mask = ts.index.year == y
            # 将当前年份 y 对应的数据存入 expected 字典，索引为年中的第几天
            expected[y] = Series(ts.values[mask], index=doy[mask])
        # 将 expected 转换为 DataFrame 格式，数据类型为 float，并转置
        expected = DataFrame(expected, dtype=float).T
        # 将 expected 的索引转换为 np.int32 类型
        expected.index = expected.index.astype(np.int32)
        # 使用测试工具 tm 进行结果比较，确保 result 等于 expected
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试按月进行数据透视表操作
    def test_monthly(self):
        # 生成一个从 "1/1/2000" 到 "12/31/2004" 的日期范围，频率为每月末
        rng = date_range("1/1/2000", "12/31/2004", freq="ME")
        # 创建一个时间序列，其值为从 0 开始的递增序列，索引为 rng
        ts = Series(np.arange(len(rng)), index=rng)

        # 对时间序列 ts 执行数据透视操作，按年作为索引，按月份作为列
        result = pivot_table(DataFrame(ts), index=ts.index.year, columns=ts.index.month)
        # 去除结果的列索引的第一层级
        result.columns = result.columns.droplevel(0)

        # 创建一个包含 ts 索引中每月的月份的数组
        month = np.asarray(ts.index.month)
        # 期望的结果初始化为空字典
        expected = {}
        # 对时间序列 ts 中每个年份进行循环
        for y in ts.index.year.unique().values:
            # 创建一个布尔掩码，表示 ts 中索引为当前年份 y 的位置
            mask = ts.index.year == y
            # 将当前年份 y 对应的数据存入 expected 字典，索引为月份
            expected[y] = Series(ts.values[mask], index=month[mask])
        # 将 expected 转换为 DataFrame 格式，数据类型为 float，并转置
        expected = DataFrame(expected, dtype=float).T
        # 将 expected 的索引转换为 np.int32 类型
        expected.index = expected.index.astype(np.int32)
        # 使用测试工具 tm 进行结果比较，确保 result 等于 expected
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试带有迭代器值的数据透视表操作
    def test_pivot_table_with_iterator_values(self, data):
        # GH 12017
        # 定义一个聚合函数字典，指定 "D" 列使用 sum 聚合，"E" 列使用 mean 聚合
        aggs = {"D": "sum", "E": "mean"}

        # 使用数据 data，按列 "A" 作为索引，将 aggs 字典中的键转换为列表作为值进行数据透视
        pivot_values_list = pivot_table(
            data, index=["A"], values=list(aggs.keys()), aggfunc=aggs
        )

        # 使用数据 data，按列 "A" 作为索引，将 aggs 字典的键直接作为值进行数据透视
        pivot_values_keys = pivot_table(
            data, index=["A"], values=aggs.keys(), aggfunc=aggs
        )
        # 使用测试工具 tm 进行结果比较，确保两次数据透视的结果相等
        tm.assert_frame_equal(pivot_values_keys, pivot_values_list)

        # 创建一个生成器，遍历 aggs 字典的键值
        agg_values_gen = (value for value in aggs)
        # 使用数据 data，按列 "A" 作为索引，使用生成器 agg_values_gen 作为值进行数据透视
        pivot_values_gen = pivot_table(
            data, index=["A"], values=agg_values_gen, aggfunc=aggs
        )
        # 使用测试工具 tm 进行结果比较，确保生成器方式的数据透视结果与列表方式相同
        tm.assert_frame_equal(pivot_values_gen, pivot_values_list)
    def test_pivot_table_margins_name_with_aggfunc_list(self):
        # 测试用例：验证带有聚合函数列表的边际名称的数据透视表生成
        margins_name = "Weekly"  # 定义边际名称为"Weekly"
        costs = DataFrame(
            {
                "item": ["bacon", "cheese", "bacon", "cheese"],  # 创建包含商品和成本的数据框
                "cost": [2.5, 4.5, 3.2, 3.3],
                "day": ["ME", "ME", "T", "T"],  # 商品的购买日期
            }
        )
        # 生成数据透视表，计算每个商品在每天的平均值和最大值
        table = costs.pivot_table(
            index="item",
            columns="day",
            margins=True,
            margins_name=margins_name,
            aggfunc=["mean", "max"],
        )
        ix = Index(["bacon", "cheese", margins_name], name="item")  # 设置结果数据框的索引
        tups = [
            ("mean", "cost", "ME"),
            ("mean", "cost", "T"),
            ("mean", "cost", margins_name),
            ("max", "cost", "ME"),
            ("max", "cost", "T"),
            ("max", "cost", margins_name),
        ]
        cols = MultiIndex.from_tuples(tups, names=[None, None, "day"])  # 设置结果数据框的列索引
        expected = DataFrame(table.values, index=ix, columns=cols)  # 创建预期的结果数据框
        tm.assert_frame_equal(table, expected)  # 断言实际生成的数据框与预期结果一致

    def test_categorical_margins(self, observed):
        # 测试用例：验证分类数据的边际计算
        df = DataFrame(
            {"x": np.arange(8), "y": np.arange(8) // 4, "z": np.arange(8) % 2}  # 创建包含分类数据的数据框
        )

        expected = DataFrame([[1.0, 2.0, 1.5], [5, 6, 5.5], [3, 4, 3.5]])  # 创建预期的结果数据框
        expected.index = Index([0, 1, "All"], name="y")  # 设置预期结果数据框的行索引
        expected.columns = Index([0, 1, "All"], name="z")  # 设置预期结果数据框的列索引

        # 生成分类数据的数据透视表，计算每个分类变量的统计量，并包括边际
        table = df.pivot_table("x", "y", "z", dropna=observed, margins=True)
        tm.assert_frame_equal(table, expected)  # 断言实际生成的数据框与预期结果一致

    def test_categorical_margins_category(self, observed):
        # 测试用例：验证将列转换为分类数据后的边际计算
        df = DataFrame(
            {"x": np.arange(8), "y": np.arange(8) // 4, "z": np.arange(8) % 2}  # 创建包含分类数据的数据框
        )

        expected = DataFrame([[1.0, 2.0, 1.5], [5, 6, 5.5], [3, 4, 3.5]])  # 创建预期的结果数据框
        expected.index = Index([0, 1, "All"], name="y")  # 设置预期结果数据框的行索引
        expected.columns = Index([0, 1, "All"], name="z")  # 设置预期结果数据框的列索引

        df.y = df.y.astype("category")  # 将y列转换为分类数据类型
        df.z = df.z.astype("category")  # 将z列转换为分类数据类型
        # 生成分类数据的数据透视表，计算每个分类变量的统计量，并包括边际，观察未定义的值
        table = df.pivot_table(
            "x", "y", "z", dropna=observed, margins=True, observed=False
        )
        tm.assert_frame_equal(table, expected)  # 断言实际生成的数据框与预期结果一致

    def test_margins_casted_to_float(self):
        # 测试用例：验证边际值被强制转换为浮点数的数据透视表生成
        df = DataFrame(
            {
                "A": [2, 4, 6, 8],
                "B": [1, 4, 5, 8],
                "C": [1, 3, 4, 6],
                "D": ["X", "X", "Y", "Y"],  # 创建包含不同类型数据的数据框
            }
        )

        # 生成数据透视表，计算每个分类变量的统计量，并包括边际
        result = pivot_table(df, index="D", margins=True)
        expected = DataFrame(
            {"A": [3.0, 7.0, 5], "B": [2.5, 6.5, 4.5], "C": [2.0, 5.0, 3.5]},  # 创建预期的结果数据框
            index=Index(["X", "Y", "All"], name="D"),  # 设置预期结果数据框的索引
        )
        tm.assert_frame_equal(result, expected)  # 断言实际生成的数据框与预期结果一致
    def test_pivot_with_categorical(self, observed, ordered):
        # gh-21370
        # 创建包含空值、"low"、"high"等元素的列表作为索引
        idx = [np.nan, "low", "high", "low", np.nan]
        # 创建包含空值、"A"、"B"等元素的列表作为列名
        col = [np.nan, "A", "B", np.nan, "A"]
        # 创建包含三列数据的DataFrame，其中"In"和"Col"列使用Categorical类型，"Val"列为1到5的连续整数
        df = DataFrame(
            {
                "In": Categorical(idx, categories=["low", "high"], ordered=ordered),
                "Col": Categorical(col, categories=["A", "B"], ordered=ordered),
                "Val": range(1, 6),
            }
        )
        # 根据指定的索引、列名和值生成透视表，使用观测到的数据(observed)
        result = df.pivot_table(
            index="In", columns="Col", values="Val", observed=observed
        )

        # 创建预期的列索引，类型为pd.CategoricalIndex，元素为"A"和"B"，有序性由参数ordered指定
        expected_cols = pd.CategoricalIndex(["A", "B"], ordered=ordered, name="Col")

        # 创建预期的DataFrame，数据为[[2.0, NaN], [NaN, 3.0]]，索引为Categorical类型，类别为["low", "high"]，有序性由参数ordered指定，列名为"In"
        expected = DataFrame(data=[[2.0, np.nan], [np.nan, 3.0]], columns=expected_cols)
        expected.index = Index(
            Categorical(["low", "high"], categories=["low", "high"], ordered=ordered),
            name="In",
        )

        # 使用测试框架验证result和expected的数据框是否相等
        tm.assert_frame_equal(result, expected)

        # 另一种情况，仅指定列名和值生成透视表，使用观测到的数据(observed)
        result = df.pivot_table(columns="Col", values="Val", observed=observed)

        # 创建预期的DataFrame，数据为[[3.5, 3.0]]，列索引为pd.CategoricalIndex，元素为"A"和"B"，有序性由参数ordered指定
        expected = DataFrame(
            data=[[3.5, 3.0]], columns=expected_cols, index=Index(["Val"])
        )

        # 使用测试框架验证result和expected的数据框是否相等
        tm.assert_frame_equal(result, expected)

    def test_categorical_aggfunc(self, observed):
        # GH 9534
        # 创建包含三列数据的DataFrame，"C1"列为字符串类型转为分类类型，"C2"列为字符串，"V"列为整数1到4
        df = DataFrame(
            {"C1": ["A", "B", "C", "C"], "C2": ["a", "a", "b", "b"], "V": [1, 2, 3, 4]}
        )
        # 将"C1"列转换为分类类型
        df["C1"] = df["C1"].astype("category")
        # 根据指定的索引、列名、聚合函数("count")生成透视表，使用观测到的数据(observed)和参数dropna
        result = df.pivot_table(
            "V",
            index="C1",
            columns="C2",
            dropna=observed,
            aggfunc="count",
            observed=False,
        )

        # 创建预期的行索引，类型为pd.CategoricalIndex，元素为"A"、"B"、"C"，有序性为False，名称为"C1"
        expected_index = pd.CategoricalIndex(
            ["A", "B", "C"], categories=["A", "B", "C"], ordered=False, name="C1"
        )
        # 创建预期的列索引，类型为Index，元素为"a"和"b"，名称为"C2"
        expected_columns = Index(["a", "b"], name="C2")
        # 创建预期的数据数组，数据为[[1, 0], [1, 0], [0, 2]]，类型为np.int64
        expected_data = np.array([[1, 0], [1, 0], [0, 2]], dtype=np.int64)
        # 创建预期的DataFrame，数据为expected_data，行索引为expected_index，列索引为expected_columns
        expected = DataFrame(
            expected_data, index=expected_index, columns=expected_columns
        )
        # 使用测试框架验证result和expected的数据框是否相等
        tm.assert_frame_equal(result, expected)
    # 测试用例：验证分类数据透视表的索引和列的顺序
    def test_categorical_pivot_index_ordering(self, observed):
        # GH 8731
        # 创建一个包含销售数据、月份和年份的数据框
        df = DataFrame(
            {
                "Sales": [100, 120, 220],
                "Month": ["January", "January", "January"],
                "Year": [2013, 2014, 2013],
            }
        )
        # 定义月份列表
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        # 将月份列转换为分类类型，并设置特定的月份顺序
        df["Month"] = df["Month"].astype("category").cat.set_categories(months)
        # 创建数据透视表，计算销售总额，按月份和年份分组
        result = df.pivot_table(
            values="Sales",
            index="Month",
            columns="Year",
            observed=observed,
            aggfunc="sum",
        )
        # 期望的列索引，包含年份信息
        expected_columns = Index([2013, 2014], name="Year", dtype="int64")
        # 期望的行索引，包含月份信息的分类索引
        expected_index = pd.CategoricalIndex(
            months, categories=months, ordered=False, name="Month"
        )
        # 期望的数据，按照指定的月份和年份组织销售数据
        expected_data = [[320, 120]] + [[0, 0]] * 11
        expected = DataFrame(
            expected_data, index=expected_index, columns=expected_columns
        )
        # 如果设置了 observed 参数为 True，则只保留预期结果中的 "January" 月份数据
        if observed:
            expected = expected.loc[["January"]]

        # 断言实际结果与预期结果是否一致
        tm.assert_frame_equal(result, expected)

    # 测试用例：验证非 Series 对象的数据透视表
    def test_pivot_table_not_series(self):
        # GH 4386
        # pivot_table 总是返回一个 DataFrame
        # 当 values 不是列表，并且 columns 为 None 且 aggfunc 不是列表的实例时
        # 创建包含列 'col1', 'col2', 'col3' 的数据框
        df = DataFrame({"col1": [3, 4, 5], "col2": ["C", "D", "E"], "col3": [1, 3, 9]})

        # 创建数据透视表，按照 'col3' 和 'col2' 进行索引，使用 'col1' 进行求和聚合
        result = df.pivot_table("col1", index=["col3", "col2"], aggfunc="sum")
        # 期望的索引，包含 'col3' 和 'col2' 的多级索引
        m = MultiIndex.from_arrays([[1, 3, 9], ["C", "D", "E"]], names=["col3", "col2"])
        # 期望的结果数据框，包含 'col1' 列的聚合值
        expected = DataFrame([3, 4, 5], index=m, columns=["col1"])

        # 断言实际结果与预期结果是否一致
        tm.assert_frame_equal(result, expected)

        # 创建数据透视表，按照 'col3' 进行索引，按照 'col2' 进行列分组，使用 'col1' 进行求和聚合
        result = df.pivot_table("col1", index="col3", columns="col2", aggfunc="sum")
        # 期望的结果数据框，按照 'col3' 和 'col2' 组织 'col1' 的求和结果
        expected = DataFrame(
            [[3, np.nan, np.nan], [np.nan, 4, np.nan], [np.nan, np.nan, 5]],
            index=Index([1, 3, 9], name="col3"),
            columns=Index(["C", "D", "E"], name="col2"),
        )

        # 断言实际结果与预期结果是否一致
        tm.assert_frame_equal(result, expected)

        # 创建数据透视表，按照 'col3' 进行索引，使用 'col1' 进行求和聚合
        result = df.pivot_table("col1", index="col3", aggfunc=["sum"])
        # 期望的多级列索引，包含聚合函数和列名
        m = MultiIndex.from_arrays([["sum"], ["col1"]])
        # 期望的结果数据框，按照 'col3' 组织 'col1' 的求和结果
        expected = DataFrame([3, 4, 5], index=Index([1, 3, 9], name="col3"), columns=m)

        # 断言实际结果与预期结果是否一致
        tm.assert_frame_equal(result, expected)
    def test_pivot_margins_name_unicode(self):
        # 定义希腊字符变量用于设置边际名称
        greek = "\u0394\u03bf\u03ba\u03b9\u03bc\u03ae"
        # 创建一个 DataFrame 对象 frame，包含一列名为 'foo' 的数据
        frame = DataFrame({"foo": [1, 2, 3]}, columns=Index(["foo"], dtype=object))
        # 调用 pivot_table 函数生成透视表 table，对 'frame' 进行操作
        table = pivot_table(
            frame, index=["foo"], aggfunc=len, margins=True, margins_name=greek
        )
        # 创建一个 Index 对象 index，包含指定的索引值和名称
        index = Index([1, 2, 3, greek], dtype="object", name="foo")
        # 期望的结果 DataFrame 对象 expected，与 table 进行比较
        expected = DataFrame(index=index, columns=[])
        # 使用测试模块 tm 来比较 table 和 expected 的一致性
        tm.assert_frame_equal(table, expected)

    def test_pivot_string_as_func(self):
        # GH #18713 GitHub 上的问题标识
        # 用于验证正确性的数据 DataFrame 对象 data，包含三列数据 A, B, C
        data = DataFrame(
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
                "C": range(11),
            }
        )

        # 使用 pivot_table 函数生成结果对象 result，根据指定索引和聚合函数进行操作
        result = pivot_table(data, index="A", columns="B", aggfunc="sum")
        # 创建 MultiIndex 对象 mi，包含指定的级别、代码和名称
        mi = MultiIndex(
            levels=[["C"], ["one", "two"]], codes=[[0, 0], [0, 1]], names=[None, "B"]
        )
        # 期望的结果 DataFrame 对象 expected，与 result 进行比较
        expected = DataFrame(
            {("C", "one"): {"bar": 15, "foo": 13}, ("C", "two"): {"bar": 7, "foo": 20}},
            columns=mi,
        ).rename_axis("A")
        # 使用测试模块 tm 来比较 result 和 expected 的一致性
        tm.assert_frame_equal(result, expected)

        # 再次使用 pivot_table 函数生成结果对象 result，根据指定索引和多个聚合函数进行操作
        result = pivot_table(data, index="A", columns="B", aggfunc=["sum", "mean"])
        # 创建 MultiIndex 对象 mi，包含指定的级别、代码和名称
        mi = MultiIndex(
            levels=[["sum", "mean"], ["C"], ["one", "two"]],
            codes=[[0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1]],
            names=[None, None, "B"],
        )
        # 期望的结果 DataFrame 对象 expected，与 result 进行比较
        expected = DataFrame(
            {
                ("mean", "C", "one"): {"bar": 5.0, "foo": 3.25},
                ("mean", "C", "two"): {"bar": 7.0, "foo": 6.666666666666667},
                ("sum", "C", "one"): {"bar": 15, "foo": 13},
                ("sum", "C", "two"): {"bar": 7, "foo": 20},
            },
            columns=mi,
        ).rename_axis("A")
        # 使用测试模块 tm 来比较 result 和 expected 的一致性
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("kwargs", [{"a": 2}, {"a": 2, "b": 3}, {"b": 3, "a": 2}])
    def test_pivot_table_kwargs(self, kwargs):
        # GH#57884
        # 定义函数 f，接受参数 x、a 和可选参数 b，默认为 3，计算并返回 x 的总和乘以 a 加上 b 的结果
        def f(x, a, b=3):
            return x.sum() * a + b

        # 定义函数 g，接受参数 x，调用函数 f 并传递参数 x 和 **kwargs，返回计算结果
        def g(x):
            return f(x, **kwargs)

        # 创建 DataFrame df，包含三列 A、B 和 X，每列包含数值和字符串数据
        df = DataFrame(
            {
                "A": ["good", "bad", "good", "bad", "good"],
                "B": ["one", "two", "one", "three", "two"],
                "X": [2, 5, 4, 20, 10],
            }
        )
        
        # 调用 pivot_table 函数，传递 DataFrame df，以 A 为行索引，B 为列索引，X 为值，使用函数 f 作为聚合函数，同时传递额外的 kwargs 参数
        result = pivot_table(
            df, index="A", columns="B", values="X", aggfunc=f, **kwargs
        )
        
        # 用相同的参数调用 pivot_table 函数，但使用函数 g 作为聚合函数
        expected = pivot_table(df, index="A", columns="B", values="X", aggfunc=g)
        
        # 使用测试工具比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "kwargs", [{}, {"b": 10}, {"a": 3}, {"a": 3, "b": 10}, {"b": 10, "a": 3}]
    )
    def test_pivot_table_kwargs_margin(self, data, kwargs):
        # GH#57884
        # 定义函数 f，接受参数 x 和可选参数 a 和 b，默认值分别为 5 和 7，计算并返回 x 的总和加上 b 后乘以 a 的结果
        def f(x, a=5, b=7):
            return (x.sum() + b) * a

        # 定义函数 g，接受参数 x，调用函数 f 并传递参数 x 和 **kwargs，返回计算结果
        def g(x):
            return f(x, **kwargs)

        # 调用 DataFrame 的 pivot_table 方法，传递参数 values="D"、index=["A", "B"]、columns="C"、aggfunc=f，以及其他参数如 margins=True 和 fill_value=0，并将 **kwargs 参数传递给函数
        result = data.pivot_table(
            values="D",
            index=["A", "B"],
            columns="C",
            aggfunc=f,
            margins=True,
            fill_value=0,
            **kwargs,
        )

        # 用相同的参数调用 DataFrame 的 pivot_table 方法，但使用函数 g 作为聚合函数
        expected = data.pivot_table(
            values="D",
            index=["A", "B"],
            columns="C",
            aggfunc=g,
            margins=True,
            fill_value=0,
        )

        # 使用测试工具比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "f, f_numpy",
        [
            ("sum", np.sum),
            ("mean", np.mean),
            ("min", np.min),
            (["sum", "mean"], [np.sum, np.mean]),
            (["sum", "min"], [np.sum, np.min]),
            (["max", "mean"], [np.max, np.mean]),
        ],
    )
    def test_pivot_string_func_vs_func(self, f, f_numpy, data):
        # GH #18713
        # 准备数据，确保列 "C" 被移除，以保持一致性
        data = data.drop(columns="C")
        
        # 调用 pivot_table 函数，传递数据 data，以 A 为行索引，B 为列索引，使用字符串函数 f 作为聚合函数
        result = pivot_table(data, index="A", columns="B", aggfunc=f)
        
        # 用相同的参数调用 pivot_table 函数，但使用对应的 numpy 函数 f_numpy 作为聚合函数
        expected = pivot_table(data, index="A", columns="B", aggfunc=f_numpy)

        # 如果 numpy 的版本低于 1.25 并且 f_numpy 是列表，则需要进行列名的映射处理
        if not np_version_gte1p25 and isinstance(f_numpy, list):
            mapper = {"amin": "min", "amax": "max", "sum": "sum", "mean": "mean"}
            expected.columns = expected.columns.map(
                lambda x: (mapper[x[0]], x[1], x[2])
            )
        
        # 使用测试工具比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.slow
    def test_pivot_number_of_levels_larger_than_int32(
        self, performance_warning, monkeypatch
    ):
        # 用于测试大于 int32 的级别数量的数据透视表性能
    ):
        # GH 20601
        # GH 26314: Change ValueError to PerformanceWarning
        # 定义一个名为MockUnstacker的类，继承自reshape_lib._Unstacker类
        class MockUnstacker(reshape_lib._Unstacker):
            def __init__(self, *args, **kwargs) -> None:
                # 在初始化方法中调用父类的初始化方法，引发异常
                super().__init__(*args, **kwargs)
                # 抛出异常，指示不要计算最终结果
                raise Exception("Don't compute final result.")

        # 使用monkeypatch上下文管理器
        with monkeypatch.context() as m:
            # 替换reshape_lib._Unstacker为MockUnstacker类
            m.setattr(reshape_lib, "_Unstacker", MockUnstacker)
            # 创建一个DataFrame对象
            df = DataFrame(
                {"ind1": np.arange(2**16), "ind2": np.arange(2**16), "count": 0}
            )

            # 定义警告消息
            msg = "The following operation may generate"
            # 断言在执行以下代码块时会产生性能警告，匹配消息msg
            with tm.assert_produces_warning(performance_warning, match=msg):
                # 断言执行以下代码块时会抛出异常，匹配消息"Don't compute final result."
                with pytest.raises(Exception, match="Don't compute final result."):
                    # 对DataFrame对象执行pivot_table操作
                    df.pivot_table(
                        index="ind1", columns="ind2", values="count", aggfunc="count"
                    )

    def test_pivot_table_aggfunc_dropna(self, dropna):
        # GH 22159
        # 创建一个DataFrame对象
        df = DataFrame(
            {
                "fruit": ["apple", "peach", "apple"],
                "size": [1, 1, 2],
                "taste": [7, 6, 6],
            }
        )

        # 定义返回1的函数
        def ret_one(x):
            return 1

        # 定义返回x数组的和的函数
        def ret_sum(x):
            return sum(x)

        # 定义返回NaN的函数
        def ret_none(x):
            return np.nan

        # 调用pivot_table函数进行数据透视
        result = pivot_table(
            df, columns="fruit", aggfunc=[ret_sum, ret_none, ret_one], dropna=dropna
        )

        # 预期的DataFrame数据
        data = [[3, 1, np.nan, np.nan, 1, 1], [13, 6, np.nan, np.nan, 1, 1]]
        # 创建多级索引列
        col = MultiIndex.from_product(
            [["ret_sum", "ret_none", "ret_one"], ["apple", "peach"]],
            names=[None, "fruit"],
        )
        expected = DataFrame(data, index=["size", "taste"], columns=col)

        # 如果dropna为True，则从列中删除NaN值
        if dropna:
            expected = expected.dropna(axis="columns")

        # 断言两个DataFrame对象是否相等
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_aggfunc_scalar_dropna(self, dropna):
        # GH 22159
        # 创建一个DataFrame对象
        df = DataFrame(
            {"A": ["one", "two", "one"], "x": [3, np.nan, 2], "y": [1, np.nan, np.nan]}
        )

        # 调用pivot_table函数进行数据透视
        result = pivot_table(df, columns="A", aggfunc="mean", dropna=dropna)

        # 预期的DataFrame数据
        data = [[2.5, np.nan], [1, np.nan]]
        col = Index(["one", "two"], name="A")
        expected = DataFrame(data, index=["x", "y"], columns=col)

        # 如果dropna为True，则从列中删除NaN值
        if dropna:
            expected = expected.dropna(axis="columns")

        # 断言两个DataFrame对象是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("margins", [True, False])
    def test_pivot_table_empty_aggfunc(self, margins):
        # GH 9186 & GH 13483 & GH 49240
        # 创建一个测试用的数据框 df，包括列 "A", "id", "C", "D"
        df = DataFrame(
            {
                "A": [2, 2, 3, 3, 2],
                "id": [5, 6, 7, 8, 9],
                "C": ["p", "q", "q", "p", "q"],
                "D": [None, None, None, None, None],
            }
        )
        # 对 df 执行 pivot_table 操作，根据列 "A" 来分组，列 "D" 作为列，"id" 列的大小作为聚合函数 np.size 的结果，同时计算边际值
        result = df.pivot_table(
            index="A", columns="D", values="id", aggfunc=np.size, margins=margins
        )
        # 创建预期的列索引对象 exp_cols
        exp_cols = Index([], name="D")
        # 创建预期的数据框 expected，具有空的行索引和列索引
        expected = DataFrame(index=Index([], dtype="int64", name="A"), columns=exp_cols)
        # 使用测试工具函数检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_no_column_raises(self):
        # GH 10326
        # 定义一个用于聚合的函数 agg，返回数组的均值 np.mean
        def agg(arr):
            return np.mean(arr)

        # 创建一个数据框 df，包含列 "X", "Y", "Z"
        df = DataFrame({"X": [0, 0, 1, 1], "Y": [0, 1, 0, 1], "Z": [10, 20, 30, 40]})
        # 使用 pytest 检查是否会引发 KeyError 异常，并匹配错误消息 "notpresent"
        with pytest.raises(KeyError, match="notpresent"):
            # 对 df 执行 pivot_table 操作，以列 "notpresent" 作为值，"X" 作为行索引，"Y" 作为列索引，聚合函数为定义的 agg 函数
            df.pivot_table("notpresent", "X", "Y", aggfunc=agg)

    def test_pivot_table_multiindex_columns_doctest_case(self):
        # The relevant characteristic is that the call
        #  to maybe_downcast_to_dtype(agged[v], data[v].dtype) in
        #  __internal_pivot_table has `agged[v]` a DataFrame instead of Series,
        #  In this case this is because agged.columns is a MultiIndex and 'v'
        #  is only indexing on its first level.
        # 创建一个数据框 df，包含列 "A", "B", "C", "D", "E"
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

        # 对 df 执行 pivot_table 操作，计算 "D" 和 "E" 列的均值、最大值和最小值，行索引为多级索引 ["A", "C"]
        table = pivot_table(
            df,
            values=["D", "E"],
            index=["A", "C"],
            aggfunc={"D": "mean", "E": ["min", "max", "mean"]},
        )

        # 创建预期的列索引对象 cols，包括 ("D", "mean"), ("E", "max"), ("E", "mean"), ("E", "min")
        cols = MultiIndex.from_tuples(
            [("D", "mean"), ("E", "max"), ("E", "mean"), ("E", "min")]
        )

        # 创建预期的行索引对象 index，包括 [("bar", "large"), ("bar", "small"), ("foo", "large"), ("foo", "small")]
        index = MultiIndex.from_tuples(
            [("bar", "large"), ("bar", "small"), ("foo", "large"), ("foo", "small")],
            names=["A", "C"],
        )

        # 创建预期的值数组 vals，包括根据计算得到的均值、最大值和最小值
        vals = np.array(
            [
                [5.5, 9.0, 7.5, 6.0],
                [5.5, 9.0, 8.5, 8.0],
                [2.0, 5.0, 4.5, 4.0],
                [2.33333333, 6.0, 4.33333333, 2.0],
            ]
        )

        # 创建预期的数据框 expected，包括预期的值、行索引和列索引
        expected = DataFrame(vals, columns=cols, index=index)

        # 将 "E" 列的最小值和最大值转换为整数类型 np.int64
        expected[("E", "min")] = expected[("E", "min")].astype(np.int64)
        expected[("E", "max")] = expected[("E", "max")].astype(np.int64)

        # 使用测试工具函数检查 table 和 expected 是否相等
        tm.assert_frame_equal(table, expected)
    def test_pivot_table_sort_false(self):
        # 根据 GitHub 问题编号 GH#39143
        df = DataFrame(
            {
                "a": ["d1", "d4", "d3"],
                "col": ["a", "b", "c"],
                "num": [23, 21, 34],
                "year": ["2018", "2018", "2019"],
            }
        )
        # 执行数据透视表操作，禁用排序
        result = df.pivot_table(
            index=["a", "col"], columns="year", values="num", aggfunc="sum", sort=False
        )
        expected = DataFrame(
            [[23, np.nan], [21, np.nan], [np.nan, 34]],
            columns=Index(["2018", "2019"], name="year"),
            index=MultiIndex.from_arrays(
                [["d1", "d4", "d3"], ["a", "b", "c"]], names=["a", "col"]
            ),
        )
        # 使用测试工具比较结果和预期输出
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_nullable_margins(self):
        # 根据 GitHub 问题编号 GH#48681
        df = DataFrame(
            {"a": "A", "b": [1, 2], "sales": Series([10, 11], dtype="Int64")}
        )
        # 执行数据透视表操作，包含空白边界
        result = df.pivot_table(index="b", columns="a", margins=True, aggfunc="sum")
        expected = DataFrame(
            [[10, 10], [11, 11], [21, 21]],
            index=Index([1, 2, "All"], name="b"),
            columns=MultiIndex.from_tuples(
                [("sales", "A"), ("sales", "All")], names=[None, "a"]
            ),
            dtype="Int64",
        )
        # 使用测试工具比较结果和预期输出
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_sort_false_with_multiple_values(self):
        df = DataFrame(
            {
                "firstname": ["John", "Michael"],
                "lastname": ["Foo", "Bar"],
                "height": [173, 182],
                "age": [47, 33],
            }
        )
        # 执行数据透视表操作，禁用排序，并处理多个数值列
        result = df.pivot_table(
            index=["lastname", "firstname"], values=["height", "age"], sort=False
        )
        expected = DataFrame(
            [[173.0, 47.0], [182.0, 33.0]],
            columns=["height", "age"],
            index=MultiIndex.from_tuples(
                [("Foo", "John"), ("Bar", "Michael")],
                names=["lastname", "firstname"],
            ),
        )
        # 使用测试工具比较结果和预期输出
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_with_margins_and_numeric_columns(self):
        # 根据 GitHub 问题编号 GH 26568
        df = DataFrame([["a", "x", 1], ["a", "y", 2], ["b", "y", 3], ["b", "z", 4]])
        df.columns = [10, 20, 30]
        # 执行数据透视表操作，包含边界和数值列
        result = df.pivot_table(
            index=10, columns=20, values=30, aggfunc="sum", fill_value=0, margins=True
        )
        expected = DataFrame([[1, 2, 0, 3], [0, 3, 4, 7], [1, 5, 4, 10]])
        expected.columns = ["x", "y", "z", "All"]
        expected.index = ["a", "b", "All"]
        expected.columns.name = 20
        expected.index.name = 10
        # 使用测试工具比较结果和预期输出
        tm.assert_frame_equal(result, expected)
    def test_pivot_ea_dtype_dropna(self, dropna):
        # GH#47477
        # 创建一个 DataFrame 包含列 'x', 'y', 'age'，其中 'age' 是一个包含整数的 Series
        df = DataFrame({"x": "a", "y": "b", "age": Series([20, 40], dtype="Int64")})
        # 调用 pivot_table 方法，根据 'x', 'y' 列进行数据透视，计算 'age' 的均值，根据 dropna 参数处理缺失值
        result = df.pivot_table(
            index="x", columns="y", values="age", aggfunc="mean", dropna=dropna
        )
        # 创建一个预期的 DataFrame，包含一个值为 30 的单元格，用于对比结果
        expected = DataFrame(
            [[30]],
            index=Index(["a"], name="x"),
            columns=Index(["b"], name="y"),
            dtype="Float64",
        )
        # 使用 assert_frame_equal 方法比较 result 和 expected，确认结果正确性
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_datetime_warning(self):
        # GH#48683
        # 创建一个 DataFrame 包含列 'a', 'b', 'date', 'sales'，其中 'date' 是一个时间戳，'sales' 是一个浮点数列表
        df = DataFrame(
            {
                "a": "A",
                "b": [1, 2],
                "date": pd.Timestamp("2019-12-31"),
                "sales": [10.0, 11],
            }
        )
        # 使用 assert_produces_warning 方法确保不会产生警告信息
        with tm.assert_produces_warning(None):
            # 调用 pivot_table 方法，根据 'b', 'date' 列进行数据透视，计算 'sales' 的总和，并创建边际汇总
            result = df.pivot_table(
                index=["b", "date"], columns="a", margins=True, aggfunc="sum"
            )
        # 创建一个预期的 DataFrame，包含与 result 结果对应的数据，用于对比
        expected = DataFrame(
            [[10.0, 10.0], [11.0, 11.0], [21.0, 21.0]],
            index=MultiIndex.from_arrays(
                [
                    Index([1, 2, "All"], name="b"),
                    Index(
                        [pd.Timestamp("2019-12-31"), pd.Timestamp("2019-12-31"), ""],
                        dtype=object,
                        name="date",
                    ),
                ]
            ),
            columns=MultiIndex.from_tuples(
                [("sales", "A"), ("sales", "All")], names=[None, "a"]
            ),
        )
        # 使用 assert_frame_equal 方法比较 result 和 expected，确认结果正确性
        tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，用于测试包含混合嵌套元组的数据透视表
    def test_pivot_table_with_mixed_nested_tuples(self):
        # GH 50342，指明这个测试是为了解决 GitHub 上的 Issue 50342
        # 创建一个 DataFrame 对象，包含多列数据
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
                ("col5",): [
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "bar",
                    "bar",
                    "bar",
                    "bar",
                ],
                ("col6", 6): [
                    "one",
                    "one",
                    "one",
                    "two",
                    "two",
                    "one",
                    "one",
                    "two",
                    "two",
                ],
                (7, "seven"): [
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
            }
        )
        # 调用 pivot_table 函数，生成透视表，聚合函数为求和，统计列为'D'，行索引为['A', 'B']，列索引为[(7, "seven")]
        result = pivot_table(
            df, values="D", index=["A", "B"], columns=[(7, "seven")], aggfunc="sum"
        )
        # 期望的透视表结果 DataFrame 对象
        expected = DataFrame(
            [[4.0, 5.0], [7.0, 6.0], [4.0, 1.0], [np.nan, 6.0]],
            # 设置列索引的名称为 (7, "seven")
            columns=Index(["large", "small"], name=(7, "seven")),
            # 设置行索引为 MultiIndex 对象，包含两层索引，名称分别为'A'和'B'
            index=MultiIndex.from_arrays(
                [["bar", "bar", "foo", "foo"], ["one", "two"] * 2], names=["A", "B"]
            ),
        )
        # 使用测试工具函数 assert_frame_equal 检查计算结果和期望结果是否一致
        tm.assert_frame_equal(result, expected)
    # 定义测试函数，用于测试数据框的透视表操作，检查不同数值情况下的唯一值聚合函数
    def test_pivot_table_aggfunc_nunique_with_different_values(self):
        # 创建一个测试用的数据框
        test = DataFrame(
            {
                "a": range(10),
                "b": range(10),
                "c": range(10),
                "d": range(10),
            }
        )

        # 创建一个多级索引，其中包含唯一值聚合函数 "nunique" 和列名 "c"，以及序号为 b 的范围为 0 到 9 的标签
        columnval = MultiIndex.from_arrays(
            [
                ["nunique" for i in range(10)],
                ["c" for i in range(10)],
                range(10),
            ],
            names=(None, None, "b"),
        )

        # 创建一个大小为 (10, 10) 的数组，所有元素初始化为 NaN，并将对角线上的元素设置为 1.0
        nparr = np.full((10, 10), np.nan)
        np.fill_diagonal(nparr, 1.0)

        # 创建预期的数据框，使用 nparr 作为数据，index 使用范围为 0 到 9 的标签 "a"，columns 使用上述创建的 columnval
        expected = DataFrame(nparr, index=Index(range(10), name="a"), columns=columnval)

        # 调用测试数据框的 pivot_table 方法，对 "a" 列进行索引，"b" 列进行列转换，"c" 列作为值，使用唯一值聚合函数 "nunique"
        result = test.pivot_table(
            index=[
                "a",
            ],
            columns=[
                "b",
            ],
            values=[
                "c",
            ],
            aggfunc=["nunique"],
        )

        # 使用测试框架中的 assert_frame_equal 方法，比较计算结果 result 和预期结果 expected 是否一致
        tm.assert_frame_equal(result, expected)
class TestPivot:
    def test_pivot(self):
        # 准备测试数据
        data = {
            "index": ["A", "B", "C", "C", "B", "A"],
            "columns": ["One", "One", "One", "Two", "Two", "Two"],
            "values": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }

        # 创建数据帧
        frame = DataFrame(data)
        
        # 执行数据透视
        pivoted = frame.pivot(index="index", columns="columns", values="values")

        # 期望的透视结果
        expected = DataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )

        # 设置期望结果的行和列名称
        expected.index.name, expected.columns.name = "index", "columns"
        
        # 断言透视结果与期望结果相同
        tm.assert_frame_equal(pivoted, expected)

        # 检查行和列的名称是否正确设置
        assert pivoted.index.name == "index"
        assert pivoted.columns.name == "columns"

        # 不指定 values 参数的情况下进行透视
        pivoted = frame.pivot(index="index", columns="columns")
        
        # 检查透视结果的行名称是否正确设置
        assert pivoted.index.name == "index"
        
        # 检查透视结果的列名称是否正确设置为空和"columns"
        assert pivoted.columns.names == (None, "columns")

    def test_pivot_duplicates(self):
        # 准备包含重复条目的测试数据
        data = DataFrame(
            {
                "a": ["bar", "bar", "foo", "foo", "foo"],
                "b": ["one", "two", "one", "one", "two"],
                "c": [1.0, 2.0, 3.0, 3.0, 4.0],
            }
        )
        
        # 断言尝试透视时会抛出值错误异常，其中包含"duplicate entries"
        with pytest.raises(ValueError, match="duplicate entries"):
            data.pivot(index="a", columns="b", values="c")

    def test_pivot_empty(self):
        # 准备空数据帧
        df = DataFrame(columns=["a", "b", "c"])
        
        # 执行空数据帧的透视
        result = df.pivot(index="a", columns="b", values="c")
        
        # 期望得到的空数据帧结果
        expected = DataFrame(index=[], columns=[])
        
        # 断言空数据帧的透视结果与期望结果相同，不检查名称
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.parametrize("dtype", [object, "string"])
    def test_pivot_integer_bug(self, dtype):
        # 准备包含整数 bug 的测试数据
        df = DataFrame(data=[("A", "1", "A1"), ("B", "2", "B2")], dtype=dtype)
        
        # 执行包含整数 bug 的透视
        result = df.pivot(index=1, columns=0, values=2)
        
        # 断言结果的列与期望的索引相等，索引包括["A", "B"]，名称为0，dtype为参数提供的类型
        tm.assert_index_equal(result.columns, Index(["A", "B"], name=0, dtype=dtype))
    def test_pivot_index_none(self):
        # GH#3962
        # 定义测试数据，包括索引、列和对应数值
        data = {
            "index": ["A", "B", "C", "C", "B", "A"],
            "columns": ["One", "One", "One", "Two", "Two", "Two"],
            "values": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }

        # 创建 DataFrame，并将 "index" 列设置为索引
        frame = DataFrame(data).set_index("index")
        
        # 对 DataFrame 进行数据透视，根据 "columns" 列和 "values" 列生成新的 DataFrame
        result = frame.pivot(columns="columns", values="values")
        
        # 定义预期的 DataFrame
        expected = DataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )

        # 设置预期 DataFrame 的索引名和列名
        expected.index.name, expected.columns.name = "index", "columns"
        
        # 使用测试工具比较结果 DataFrame 和预期 DataFrame
        tm.assert_frame_equal(result, expected)

        # 在省略 "values" 的情
    def test_pivot_columns_is_none(self):
        # GH#48293
        # 创建一个包含空列名的DataFrame，用于测试pivot(columns=None)方法
        df = DataFrame({None: [1], "b": 2, "c": 3})
        # 调用pivot方法，以空值作为列名，生成结果DataFrame
        result = df.pivot(columns=None)
        # 期望的结果DataFrame
        expected = DataFrame({("b", 1): [2], ("c", 1): 3})
        # 检查结果DataFrame是否与期望DataFrame相等
        tm.assert_frame_equal(result, expected)

        # 再次调用pivot方法，此次指定index为"b"
        result = df.pivot(columns=None, index="b")
        # 期望的结果DataFrame
        expected = DataFrame({("c", 1): 3}, index=Index([2], name="b"))
        # 检查结果DataFrame是否与期望DataFrame相等
        tm.assert_frame_equal(result, expected)

        # 第三次调用pivot方法，指定index为"b"，values为"c"
        result = df.pivot(columns=None, index="b", values="c")
        # 期望的结果DataFrame
        expected = DataFrame({1: 3}, index=Index([2], name="b"))
        # 检查结果DataFrame是否与期望DataFrame相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="None is cast to NaN")
    def test_pivot_index_is_none(self):
        # GH#48293
        # 创建一个包含空列名的DataFrame，用于测试pivot(columns="b", index=None)方法
        df = DataFrame({None: [1], "b": 2, "c": 3})

        # 调用pivot方法，指定columns为"b"，index为None
        result = df.pivot(columns="b", index=None)
        # 期望的结果DataFrame
        expected = DataFrame({("c", 2): 3}, index=[1])
        expected.columns.names = [None, "b"]
        # 检查结果DataFrame是否与期望DataFrame相等
        tm.assert_frame_equal(result, expected)

        # 再次调用pivot方法，指定columns为"b"，index为None，values为"c"
        result = df.pivot(columns="b", index=None, values="c")
        # 期望的结果DataFrame
        expected = DataFrame(3, index=[1], columns=Index([2], name="b"))
        # 检查结果DataFrame是否与期望DataFrame相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="None is cast to NaN")
    def test_pivot_values_is_none(self):
        # GH#48293
        # 创建一个包含空列名的DataFrame，用于测试pivot(columns="b", index="c", values=None)方法
        df = DataFrame({None: [1], "b": 2, "c": 3})

        # 调用pivot方法，指定columns为"b"，index为"c"，values为None
        result = df.pivot(columns="b", index="c", values=None)
        # 期望的结果DataFrame
        expected = DataFrame(
            1, index=Index([3], name="c"), columns=Index([2], name="b")
        )
        # 检查结果DataFrame是否与期望DataFrame相等
        tm.assert_frame_equal(result, expected)

        # 再次调用pivot方法，指定columns为"b"，values为None
        result = df.pivot(columns="b", values=None)
        # 期望的结果DataFrame
        expected = DataFrame(1, index=[0], columns=Index([2], name="b"))
        # 检查结果DataFrame是否与期望DataFrame相等
        tm.assert_frame_equal(result, expected)

    def test_pivot_not_changing_index_name(self):
        # GH#52692
        # 创建一个DataFrame，用于测试pivot方法不改变index名字的情况
        df = DataFrame({"one": ["a"], "two": 0, "three": 1})
        # 复制一份预期的DataFrame
        expected = df.copy(deep=True)
        # 调用pivot方法，指定index为"one"，columns为"two"，values为"three"
        df.pivot(index="one", columns="two", values="three")
        # 检查DataFrame是否与预期DataFrame相等
        tm.assert_frame_equal(df, expected)

    def test_pivot_table_empty_dataframe_correct_index(self):
        # GH 21932
        # 创建一个空DataFrame，用于测试pivot_table方法
        df = DataFrame([], columns=["a", "b", "value"])
        # 调用pivot_table方法，指定index为"a"，columns为"b"，values为"value"，aggfunc为"count"
        pivot = df.pivot_table(index="a", columns="b", values="value", aggfunc="count")

        # 期望的结果Index
        expected = Index([], dtype="object", name="b")
        # 检查pivot的columns是否与预期Index相等
        tm.assert_index_equal(pivot.columns, expected)
    def test_pivot_table_handles_explicit_datetime_types(self):
        # 测试函数：测试数据透视表处理显式日期时间类型的情况
        df = DataFrame(
            [
                {"a": "x", "date_str": "2023-01-01", "amount": 1},
                {"a": "y", "date_str": "2023-01-02", "amount": 2},
                {"a": "z", "date_str": "2023-01-03", "amount": 3},
            ]
        )
        # 将日期字符串转换为日期时间类型列
        df["date"] = pd.to_datetime(df["date_str"])

        # 禁止产生警告的上下文环境
        with tm.assert_produces_warning(False):
            # 创建数据透视表，按索引 ['a', 'date']、值 ['amount'] 进行汇总求和，包括边距
            pivot = df.pivot_table(
                index=["a", "date"], values=["amount"], aggfunc="sum", margins=True
            )

        # 期望的多级索引
        expected = MultiIndex.from_tuples(
            [
                ("x", datetime.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")),
                ("y", datetime.strptime("2023-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")),
                ("z", datetime.strptime("2023-01-03 00:00:00", "%Y-%m-%d %H:%M:%S")),
                ("All", ""),
            ],
            names=["a", "date"],
        )
        # 断言数据透视表的索引与期望的索引相等
        tm.assert_index_equal(pivot.index, expected)

    def test_pivot_table_with_margins_and_numeric_column_names(self):
        # 测试函数：测试带有边距和数值列名的数据透视表
        df = DataFrame([["a", "x", 1], ["a", "y", 2], ["b", "y", 3], ["b", "z", 4]])

        # 创建数据透视表，按索引 0、列名 1、值 2 进行汇总求和，填充缺失值为 0，包括边距
        result = df.pivot_table(
            index=0, columns=1, values=2, aggfunc="sum", fill_value=0, margins=True
        )

        # 期望的数据透视表结果
        expected = DataFrame(
            [[1, 2, 0, 3], [0, 3, 4, 7], [1, 5, 4, 10]],
            columns=Index(["x", "y", "z", "All"], name=1),
            index=Index(["a", "b", "All"], name=0),
        )
        # 断言数据透视表的结果与期望的结果相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("m", [1, 10])
    def test_unstack_copy(self, m):
        # 测试函数：测试 unstack 操作的拷贝行为
        levels = np.arange(m)
        index = MultiIndex.from_product([levels] * 2)
        values = np.arange(m * m * 100).reshape(m * m, 100)
        df = DataFrame(values, index, np.arange(100))
        df_orig = df.copy()
        # 对 DataFrame 执行 unstack 操作，不进行排序
        result = df.unstack(sort=False)
        # 修改结果的第一个元素
        result.iloc[0, 0] = -1
        # 断言原始 DataFrame 与未改变的 DataFrame 相等
        tm.assert_frame_equal(df, df_orig)
```