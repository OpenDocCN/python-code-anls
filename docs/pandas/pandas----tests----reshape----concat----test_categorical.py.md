# `D:\src\scipysrc\pandas\pandas\tests\reshape\concat\test_categorical.py`

```
    from datetime import datetime  # 导入datetime模块中的datetime类，用于处理日期和时间

    import numpy as np  # 导入NumPy库，用于处理数组和数值计算

    from pandas.core.dtypes.dtypes import CategoricalDtype  # 从pandas核心数据类型中导入CategoricalDtype类，用于处理分类数据类型

    import pandas as pd  # 导入Pandas库，用于数据分析和处理
    from pandas import (  # 从Pandas库中导入以下类和函数
        Categorical,  # 用于处理分类数据类型的类
        DataFrame,  # 用于处理表格数据的类
        Series,  # 用于处理一维数据的类
    )
    import pandas._testing as tm  # 导入Pandas的测试工具模块作为tm，用于测试辅助函数和方法


    class TestCategoricalConcat:  # 定义测试类TestCategoricalConcat

        def test_categorical_concat(self, sort):  # 定义测试方法test_categorical_concat，接受sort参数
            # 查看GitHub issue 10177
            df1 = DataFrame(  # 创建DataFrame df1
                np.arange(18, dtype="int64").reshape(6, 3), columns=["a", "b", "c"]
            )

            df2 = DataFrame(  # 创建DataFrame df2
                np.arange(14, dtype="int64").reshape(7, 2), columns=["a", "c"]
            )

            cat_values = ["one", "one", "two", "one", "two", "two", "one"]  # 创建分类数据列表cat_values
            df2["h"] = Series(Categorical(cat_values))  # 将cat_values转换为分类类型Series，并赋给df2的新列"h"

            res = pd.concat((df1, df2), axis=0, ignore_index=True, sort=sort)  # 使用pd.concat将df1和df2按行连接，返回结果赋给res
            exp = DataFrame(  # 创建期望的DataFrame exp
                {
                    "a": [0, 3, 6, 9, 12, 15, 0, 2, 4, 6, 8, 10, 12],  # 设置列"a"的期望值
                    "b": [
                        1,
                        4,
                        7,
                        10,
                        13,
                        16,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],  # 设置列"b"的期望值
                    "c": [2, 5, 8, 11, 14, 17, 1, 3, 5, 7, 9, 11, 13],  # 设置列"c"的期望值
                    "h": [None] * 6 + cat_values,  # 设置列"h"的期望值，前6行为None，后面按cat_values填充
                }
            )
            exp["h"] = exp["h"].astype(df2["h"].dtype)  # 将exp中"h"列的数据类型转换为df2中"h"列的数据类型
            tm.assert_frame_equal(res, exp)  # 使用测试工具tm的assert_frame_equal方法比较res和exp是否相等

        def test_categorical_concat_dtypes(self, using_infer_string):  # 定义测试方法test_categorical_concat_dtypes，接受using_infer_string参数
            # GitHub issue 8143
            index = ["cat", "obj", "num"]  # 创建索引列表index
            cat = Categorical(["a", "b", "c"])  # 创建分类类型数据cat
            obj = Series(["a", "b", "c"])  # 创建一维数据obj
            num = Series([1, 2, 3])  # 创建一维数据num
            df = pd.concat([Series(cat), obj, num], axis=1, keys=index)  # 使用pd.concat按列连接cat、obj和num，并指定keys为index

            result = df.dtypes == (  # 检查DataFrame df的每列数据类型是否等于下列条件
                object if not using_infer_string else "string[pyarrow_numpy]"
            )
            expected = Series([False, True, False], index=index)  # 创建期望的数据类型Series expected
            tm.assert_series_equal(result, expected)  # 使用测试工具tm的assert_series_equal方法比较result和expected是否相等

            result = df.dtypes == "int64"  # 检查DataFrame df的每列数据类型是否都是int64
            expected = Series([False, False, True], index=index)  # 创建期望的数据类型Series expected
            tm.assert_series_equal(result, expected)  # 使用测试工具tm的assert_series_equal方法比较result和expected是否相等

            result = df.dtypes == "category"  # 检查DataFrame df的每列数据类型是否都是category
            expected = Series([True, False, False], index=index)  # 创建期望的数据类型Series expected
            tm.assert_series_equal(result, expected)  # 使用测试工具tm的assert_series_equal方法比较result和expected是否相等
    def test_concat_categoricalindex(self):
        # GH 16111, categories that aren't lexsorted
        # 定义一个不是按字典序排列的类别列表
        categories = [9, 0, 1, 2, 3]

        # 创建 Series a，使用非字典序排列的类别作为索引
        a = Series(1, index=pd.CategoricalIndex([9, 0], categories=categories))
        # 创建 Series b，使用非字典序排列的类别作为索引
        b = Series(2, index=pd.CategoricalIndex([0, 1], categories=categories))
        # 创建 Series c，使用非字典序排列的类别作为索引
        c = Series(3, index=pd.CategoricalIndex([1, 2], categories=categories))

        # 将 Series a、b、c 沿轴 1 进行拼接
        result = pd.concat([a, b, c], axis=1)

        # 期望的索引，使用了全部类别的字典序排列
        exp_idx = pd.CategoricalIndex([9, 0, 1, 2], categories=categories)
        # 期望的 DataFrame，包含了拼接后的数据和列信息
        exp = DataFrame(
            {
                0: [1, 1, np.nan, np.nan],
                1: [np.nan, 2, 2, np.nan],
                2: [np.nan, np.nan, 3, 3],
            },
            columns=[0, 1, 2],
            index=exp_idx,
        )
        # 使用断言函数验证结果与期望是否一致
        tm.assert_frame_equal(result, exp)

    def test_categorical_concat_preserve(self):
        # GH 8641  series concat not preserving category dtype
        # GH 13524 can concat different categories
        # 创建一个 category 类型的 Series s 和 s2
        s = Series(list("abc"), dtype="category")
        s2 = Series(list("abd"), dtype="category")

        # 期望的 Series，包含拼接后的数据
        exp = Series(list("abcabd"))
        # 使用 concat 函数拼接 s 和 s2
        res = pd.concat([s, s2], ignore_index=True)
        # 使用断言函数验证结果与期望是否一致
        tm.assert_series_equal(res, exp)

        # 期望的 Series，包含拼接后的数据和保持的 category 类型
        exp = Series(list("abcabc"), dtype="category")
        # 使用 concat 函数拼接 s 和 s，并保持 category 类型
        res = pd.concat([s, s], ignore_index=True)
        # 使用断言函数验证结果与期望是否一致
        tm.assert_series_equal(res, exp)

        # 期望的 Series，包含拼接后的数据和指定的索引
        exp = Series(list("abcabc"), index=[0, 1, 2, 0, 1, 2], dtype="category")
        # 使用 concat 函数拼接 s 和 s，并使用默认索引
        res = pd.concat([s, s])
        # 使用断言函数验证结果与期望是否一致
        tm.assert_series_equal(res, exp)

        # 创建 Series a 和 b，分别包含整数和 category 类型的数据
        a = Series(np.arange(6, dtype="int64"))
        b = Series(list("aabbca"))

        # 创建 DataFrame df2，包含整数和 category 类型的列
        df2 = DataFrame({"A": a, "B": b.astype(CategoricalDtype(list("cab")))})
        # 使用 concat 函数拼接 df2 和 df2
        res = pd.concat([df2, df2])
        # 期望的 DataFrame，包含拼接后的数据和保持的 category 类型
        exp = DataFrame(
            {
                "A": pd.concat([a, a]),
                "B": pd.concat([b, b]).astype(CategoricalDtype(list("cab"))),
            }
        )
        # 使用断言函数验证结果与期望是否一致
        tm.assert_frame_equal(res, exp)

    def test_categorical_index_preserver(self):
        # 创建 Series a 和 b，分别包含整数和 category 类型的数据
        a = Series(np.arange(6, dtype="int64"))
        b = Series(list("aabbca"))

        # 创建 DataFrame df2，包含整数和 category 类型的列，并将 B 列设置为索引
        df2 = DataFrame({"A": a, "B": b.astype(CategoricalDtype(list("cab")))}).set_index("B")
        # 使用 concat 函数拼接 df2 和 df2
        result = pd.concat([df2, df2])
        # 期望的 DataFrame，包含拼接后的数据和保持的 category 类型，并将 B 列设置为索引
        expected = DataFrame(
            {
                "A": pd.concat([a, a]),
                "B": pd.concat([b, b]).astype(CategoricalDtype(list("cab"))),
            }
        ).set_index("B")
        # 使用断言函数验证结果与期望是否一致
        tm.assert_frame_equal(result, expected)

        # 创建 DataFrame df3，包含整数和错误的 category 类型的列，并将 B 列设置为索引
        df3 = DataFrame({"A": a, "B": Categorical(b, categories=list("abe"))}).set_index("B")
        # 使用 concat 函数拼接 df2 和 df3
        result = pd.concat([df2, df3])
        # 期望的 DataFrame，使用 concat_compat 处理不同类别，并将索引转换为 object 类型
        expected = pd.concat(
            [
                df2.set_axis(df2.index.astype(object), axis=0),
                df3.set_axis(df3.index.astype(object), axis=0),
            ]
        )
        # 使用断言函数验证结果与期望是否一致
        tm.assert_frame_equal(result, expected)
    def test_concat_categorical_tz(self):
        # GH-23816
        # 创建包含日期范围的时间序列，设定时区为"US/Pacific"
        a = Series(pd.date_range("2017-01-01", periods=2, tz="US/Pacific"))
        # 创建包含字符串的分类数据序列
        b = Series(["a", "b"], dtype="category")
        # 将两个序列按行连接，忽略索引重置
        result = pd.concat([a, b], ignore_index=True)
        # 创建预期的时间序列和分类数据的合并结果
        expected = Series(
            [
                pd.Timestamp("2017-01-01", tz="US/Pacific"),
                pd.Timestamp("2017-01-02", tz="US/Pacific"),
                "a",
                "b",
            ]
        )
        # 断言结果序列与预期序列相等
        tm.assert_series_equal(result, expected)

    def test_concat_categorical_datetime(self):
        # GH-39443
        # 创建包含日期时间数据的DataFrame，作为分类数据类型
        df1 = DataFrame(
            {"x": Series(datetime(2021, 1, 1), index=[0], dtype="category")}
        )
        df2 = DataFrame(
            {"x": Series(datetime(2021, 1, 2), index=[1], dtype="category")}
        )

        # 将两个DataFrame按行连接
        result = pd.concat([df1, df2])
        # 创建预期的DataFrame，包含合并后的日期时间数据
        expected = DataFrame(
            {"x": Series([datetime(2021, 1, 1), datetime(2021, 1, 2)])}
        )

        # 断言结果DataFrame与预期DataFrame相等
        tm.assert_equal(result, expected)

    def test_concat_categorical_unchanged(self):
        # GH-12007
        # 测试修复分类数据与浮点数连接时的问题
        df = DataFrame(Series(["a", "b", "c"], dtype="category", name="A"))
        ser = Series([0, 1, 2], index=[0, 1, 3], name="B")
        # 将DataFrame和Series按列连接
        result = pd.concat([df, ser], axis=1)
        # 创建预期的DataFrame，包含合并后的分类数据和浮点数数据
        expected = DataFrame(
            {
                "A": Series(["a", "b", "c", np.nan], dtype="category"),
                "B": Series([0, 1, np.nan, 2], dtype="float"),
            }
        )
        # 断言结果DataFrame与预期DataFrame相等
        tm.assert_equal(result, expected)

    def test_categorical_concat_gh7864(self):
        # GH 7864
        # 确保保留分类数据的顺序
        df = DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": list("abbaae")})
        df["grade"] = Categorical(df["raw_grade"])
        df["grade"].cat.set_categories(["e", "a", "b"])

        df1 = df[0:3]
        df2 = df[3:]

        # 断言合并前后的分类数据顺序不变
        tm.assert_index_equal(df["grade"].cat.categories, df1["grade"].cat.categories)
        tm.assert_index_equal(df["grade"].cat.categories, df2["grade"].cat.categories)

        # 将两个DataFrame按行连接，并断言合并后的分类数据顺序不变
        dfx = pd.concat([df1, df2])
        tm.assert_index_equal(df["grade"].cat.categories, dfx["grade"].cat.categories)

        # 使用非公开方法'_append'将两个DataFrame连接，并断言合并后的分类数据顺序不变
        dfa = df1._append(df2)
        tm.assert_index_equal(df["grade"].cat.categories, dfa["grade"].cat.categories)
    def test_categorical_index_upcast(self):
        # GH 17629
        # 测试在合并具有非相同类别的分类索引时向对象类型的转换

        a = DataFrame({"foo": [1, 2]}, index=Categorical(["foo", "bar"]))
        b = DataFrame({"foo": [4, 3]}, index=Categorical(["baz", "bar"]))

        res = pd.concat([a, b])
        exp = DataFrame({"foo": [1, 2, 4, 3]}, index=["foo", "bar", "baz", "bar"])

        tm.assert_equal(res, exp)

        a = Series([1, 2], index=Categorical(["foo", "bar"]))
        b = Series([4, 3], index=Categorical(["baz", "bar"]))

        res = pd.concat([a, b])
        exp = Series([1, 2, 4, 3], index=["foo", "bar", "baz", "bar"])

        tm.assert_equal(res, exp)

    def test_categorical_missing_from_one_frame(self):
        # GH 25412
        # 测试一个数据框中缺失的分类

        df1 = DataFrame({"f1": [1, 2, 3]})
        df2 = DataFrame({"f1": [2, 3, 1], "f2": Series([4, 4, 4]).astype("category")})
        result = pd.concat([df1, df2], sort=True)
        dtype = CategoricalDtype([4])
        expected = DataFrame(
            {
                "f1": [1, 2, 3, 2, 3, 1],
                "f2": Categorical.from_codes([-1, -1, -1, 0, 0, 0], dtype=dtype),
            },
            index=[0, 1, 2, 0, 1, 2],
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_categorical_same_categories_different_order(self):
        # https://github.com/pandas-dev/pandas/issues/24845
        # 测试具有相同类别但顺序不同的分类合并

        c1 = pd.CategoricalIndex(["a", "a"], categories=["a", "b"], ordered=False)
        c2 = pd.CategoricalIndex(["b", "b"], categories=["b", "a"], ordered=False)
        c3 = pd.CategoricalIndex(
            ["a", "a", "b", "b"], categories=["a", "b"], ordered=False
        )

        df1 = DataFrame({"A": [1, 2]}, index=c1)
        df2 = DataFrame({"A": [3, 4]}, index=c2)

        result = pd.concat((df1, df2))
        expected = DataFrame({"A": [1, 2, 3, 4]}, index=c3)
        tm.assert_frame_equal(result, expected)
```