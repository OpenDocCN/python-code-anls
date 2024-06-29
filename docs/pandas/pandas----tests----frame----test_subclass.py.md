# `D:\src\scipysrc\pandas\pandas\tests\frame\test_subclass.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 Pytest 库，用于单元测试

import pandas as pd  # 导入 Pandas 库，并起别名为 pd
from pandas import (  # 从 Pandas 中导入以下类和函数
    DataFrame,  # 数据框类
    Index,  # 索引类
    MultiIndex,  # 多重索引类
    Series,  # 系列类
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

# 设定 Pytest 标记，忽略特定警告
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
)


class TestDataFrameSubclassing:
    def test_no_warning_on_mgr(self):
        # GH#57032，参考 GitHub 问题编号 57032
        df = tm.SubclassedDataFrame(
            {"X": [1, 2, 3], "Y": [1, 2, 3]}, index=["a", "b", "c"]
        )
        with tm.assert_produces_warning(None):
            # df.isna() 经过 _constructor_from_mgr 方法，我们希望不传递 Manager 到 __init__
            df.isna()
            df["X"].isna()

    def test_frame_subclassing_and_slicing(self):
        # 子类化数据框并确保在对其进行切片时返回正确的类
        # 参考 Pull Request 9632

        class CustomSeries(Series):
            @property
            def _constructor(self):
                return CustomSeries

            def custom_series_function(self):
                return "OK"

        class CustomDataFrame(DataFrame):
            """
            子类化 Pandas 数据框，填充模拟结果，添加自定义绘图函数。
            """

            def __init__(self, *args, **kw) -> None:
                super().__init__(*args, **kw)

            @property
            def _constructor(self):
                return CustomDataFrame

            _constructor_sliced = CustomSeries

            def custom_frame_function(self):
                return "OK"

        data = {"col1": range(10), "col2": range(10)}
        cdf = CustomDataFrame(data)

        # 我们是否得到了我们自己的数据框类？
        assert isinstance(cdf, CustomDataFrame)

        # 在选择列后，我们是否得到了我们自己的系列类？
        cdf_series = cdf.col1
        assert isinstance(cdf_series, CustomSeries)
        assert cdf_series.custom_series_function() == "OK"

        # 在按行切片后，我们是否得到了我们自己的数据框类？
        cdf_rows = cdf[1:5]
        assert isinstance(cdf_rows, CustomDataFrame)
        assert cdf_rows.custom_frame_function() == "OK"

        # 确保多重索引数据框切片后的部分是自定义类
        mcol = MultiIndex.from_tuples([("A", "A"), ("A", "B")])
        cdf_multi = CustomDataFrame([[0, 1], [2, 3]], columns=mcol)
        assert isinstance(cdf_multi["A"], CustomDataFrame)

        mcol = MultiIndex.from_tuples([("A", ""), ("B", "")])
        cdf_multi2 = CustomDataFrame([[0, 1], [2, 3]], columns=mcol)
        assert isinstance(cdf_multi2["A"], CustomSeries)
    # 测试自定义的 SubclassedDataFrame 类的元数据功能
    def test_dataframe_metadata(self):
        # 创建一个自定义的 SubclassedDataFrame 对象
        df = tm.SubclassedDataFrame(
            {"X": [1, 2, 3], "Y": [1, 2, 3]}, index=["a", "b", "c"]
        )
        # 给对象添加一个属性
        df.testattr = "XXX"

        # 断言对象的属性值
        assert df.testattr == "XXX"
        assert df[["X"]].testattr == "XXX"
        assert df.loc[["a", "b"], :].testattr == "XXX"
        assert df.iloc[[0, 1], :].testattr == "XXX"

        # 见 gh-9776
        assert df.iloc[0:1, :].testattr == "XXX"

        # 见 gh-10553
        # 对象序列化和反序列化，然后比较结果
        unpickled = tm.round_trip_pickle(df)
        tm.assert_frame_equal(df, unpickled)
        assert df._metadata == unpickled._metadata
        assert df.testattr == unpickled.testattr

    # 测试切片索引功能
    def test_indexing_sliced(self):
        # GH 11559
        # 创建一个自定义的 SubclassedDataFrame 对象
        df = tm.SubclassedDataFrame(
            {"X": [1, 2, 3], "Y": [4, 5, 6], "Z": [7, 8, 9]}, index=["a", "b", "c"]
        )
        # 使用 loc 方法进行列索引
        res = df.loc[:, "X"]
        exp = tm.SubclassedSeries([1, 2, 3], index=list("abc"), name="X")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

        # 使用 iloc 方法进行列索引
        res = df.iloc[:, 1]
        exp = tm.SubclassedSeries([4, 5, 6], index=list("abc"), name="Y")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

        # 使用 loc 方法进行列索引
        res = df.loc[:, "Z"]
        exp = tm.SubclassedSeries([7, 8, 9], index=list("abc"), name="Z")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

        # 使用 loc 方法进行行索引
        res = df.loc["a", :]
        exp = tm.SubclassedSeries([1, 4, 7], index=list("XYZ"), name="a")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

        # 使用 iloc 方法进行行索引
        res = df.iloc[1, :]
        exp = tm.SubclassedSeries([2, 5, 8], index=list("XYZ"), name="b")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

        # 使用 loc 方法进行行索引
        res = df.loc["c", :]
        exp = tm.SubclassedSeries([3, 6, 9], index=list("XYZ"), name="c")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

    # 测试子类属性错误传播
    def test_subclass_attr_err_propagation(self):
        # GH 11808
        # 定义一个继承自 DataFrame 的子类 A
        class A(DataFrame):
            @property
            def nonexistence(self):
                return self.i_dont_exist

        # 断言属性错误的异常传播
        with pytest.raises(AttributeError, match=".*i_dont_exist.*"):
            A().nonexistence
    # 定义测试方法，测试子类化的DataFrame对齐操作
    def test_subclass_align(self):
        # GH 12983: GitHub上的issue编号
        # 创建第一个子类化的DataFrame对象df1，包含两列数据和指定索引
        df1 = tm.SubclassedDataFrame(
            {"a": [1, 3, 5], "b": [1, 3, 5]}, index=list("ACE")
        )
        # 创建第二个子类化的DataFrame对象df2，包含两列数据和指定索引
        df2 = tm.SubclassedDataFrame(
            {"c": [1, 2, 4], "d": [1, 2, 4]}, index=list("ABD")
        )

        # 执行DataFrame的对齐操作，沿着axis=0轴对齐
        res1, res2 = df1.align(df2, axis=0)
        # 期望的第一个对齐后的DataFrame对象exp1
        exp1 = tm.SubclassedDataFrame(
            {"a": [1, np.nan, 3, np.nan, 5], "b": [1, np.nan, 3, np.nan, 5]},
            index=list("ABCDE"),
        )
        # 期望的第二个对齐后的DataFrame对象exp2
        exp2 = tm.SubclassedDataFrame(
            {"c": [1, 2, np.nan, 4, np.nan], "d": [1, 2, np.nan, 4, np.nan]},
            index=list("ABCDE"),
        )
        # 断言对齐后的结果res1是SubclassedDataFrame类型，并且与exp1相等
        assert isinstance(res1, tm.SubclassedDataFrame)
        tm.assert_frame_equal(res1, exp1)
        # 断言对齐后的结果res2是SubclassedDataFrame类型，并且与exp2相等
        assert isinstance(res2, tm.SubclassedDataFrame)
        tm.assert_frame_equal(res2, exp2)

        # 对Series进行单列对齐操作
        res1, res2 = df1.a.align(df2.c)
        # 断言对齐后的结果res1是SubclassedSeries类型，并且与exp1的'a'列相等
        assert isinstance(res1, tm.SubclassedSeries)
        tm.assert_series_equal(res1, exp1.a)
        # 断言对齐后的结果res2是SubclassedSeries类型，并且与exp2的'c'列相等
        assert isinstance(res2, tm.SubclassedSeries)
        tm.assert_series_equal(res2, exp2.c)

    # 定义测试方法，测试子类化的DataFrame与Series的各种对齐组合操作
    def test_subclass_align_combinations(self):
        # GH 12983: GitHub上的issue编号
        # 创建子类化的DataFrame对象df，包含两列数据和指定索引
        df = tm.SubclassedDataFrame({"a": [1, 3, 5], "b": [1, 3, 5]}, index=list("ACE"))
        # 创建子类化的Series对象s，包含一列数据和指定索引
        s = tm.SubclassedSeries([1, 2, 4], index=list("ABD"), name="x")

        # DataFrame与Series进行轴向对齐操作
        res1, res2 = df.align(s, axis=0)
        # 期望的第一个对齐后的DataFrame对象exp1
        exp1 = tm.SubclassedDataFrame(
            {"a": [1, np.nan, 3, np.nan, 5], "b": [1, np.nan, 3, np.nan, 5]},
            index=list("ABCDE"),
        )
        # 期望的第二个对齐后的Series对象exp2，注意'name'属性会丢失
        exp2 = tm.SubclassedSeries(
            [1, 2, np.nan, 4, np.nan], index=list("ABCDE"), name="x"
        )

        # 断言对齐后的结果res1是SubclassedDataFrame类型，并且与exp1相等
        assert isinstance(res1, tm.SubclassedDataFrame)
        tm.assert_frame_equal(res1, exp1)
        # 断言对齐后的结果res2是SubclassedSeries类型，并且与exp2相等
        assert isinstance(res2, tm.SubclassedSeries)
        tm.assert_series_equal(res2, exp2)

        # Series与DataFrame进行对齐操作
        res1, res2 = s.align(df)
        # 断言对齐后的结果res1是SubclassedSeries类型，并且与exp2相等
        assert isinstance(res1, tm.SubclassedSeries)
        tm.assert_series_equal(res1, exp2)
        # 断言对齐后的结果res2是SubclassedDataFrame类型，并且与exp1相等
        assert isinstance(res2, tm.SubclassedDataFrame)
        tm.assert_frame_equal(res2, exp1)

    # 定义测试方法，测试子类化的DataFrame的iterrows方法
    def test_subclass_iterrows(self):
        # GH 13977: GitHub上的issue编号
        # 创建子类化的DataFrame对象df，包含一列数据
        df = tm.SubclassedDataFrame({"a": [1]})
        # 使用iterrows方法遍历DataFrame的每一行
        for i, row in df.iterrows():
            # 断言每一行返回的row是SubclassedSeries类型，并且与df中对应索引的行相等
            assert isinstance(row, tm.SubclassedSeries)
            tm.assert_series_equal(row, df.loc[i])

    # 定义测试方法，测试子类化的DataFrame的stack方法
    def test_subclass_stack(self):
        # GH 15564: GitHub上的issue编号
        # 创建子类化的DataFrame对象df，包含3行3列数据，指定索引和列名
        df = tm.SubclassedDataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            index=["a", "b", "c"],
            columns=["X", "Y", "Z"],
        )

        # 执行stack方法，将DataFrame转换为Series
        res = df.stack()
        # 期望的stack后的Series对象exp，使用MultiIndex来表示多层索引
        exp = tm.SubclassedSeries(
            [1, 2, 3, 4, 5, 6, 7, 8, 9], index=[list("aaabbbccc"), list("XYZXYZXYZ")]
        )

        # 断言stack后的结果res与exp相等
        tm.assert_series_equal(res, exp)
    # 定义一个测试方法，用于测试多重继承的子类化DataFrame的堆叠操作
    def test_subclass_stack_multi(self):
        # GH 15564
        # 创建一个SubclassedDataFrame对象df，包含给定的数据和多重索引
        df = tm.SubclassedDataFrame(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]],
            index=MultiIndex.from_tuples(
                list(zip(list("AABB"), list("cdcd"))), names=["aaa", "ccc"]
            ),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWXX"), list("yzyz"))), names=["www", "yyy"]
            ),
        )

        # 期望的SubclassedDataFrame对象exp，用于验证堆叠操作后的结果
        exp = tm.SubclassedDataFrame(
            [
                [10, 12],
                [11, 13],
                [20, 22],
                [21, 23],
                [30, 32],
                [31, 33],
                [40, 42],
                [41, 43],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AAAABBBB"), list("ccddccdd"), list("yzyzyzyz"))),
                names=["aaa", "ccc", "yyy"],
            ),
            columns=Index(["W", "X"], name="www"),
        )

        # 对df进行堆叠操作，结果存储在变量res中，并与exp进行比较验证
        res = df.stack()
        tm.assert_frame_equal(res, exp)

        # 对df进行特定级别（"yyy"）的堆叠操作，结果存储在变量res中，并与exp进行比较验证
        res = df.stack("yyy")
        tm.assert_frame_equal(res, exp)

        # 更新期望的SubclassedDataFrame对象exp，用于特定级别（"www"）的堆叠操作验证
        exp = tm.SubclassedDataFrame(
            [
                [10, 11],
                [12, 13],
                [20, 21],
                [22, 23],
                [30, 31],
                [32, 33],
                [40, 41],
                [42, 43],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AAAABBBB"), list("ccddccdd"), list("WXWXWXWX"))),
                names=["aaa", "ccc", "www"],
            ),
            columns=Index(["y", "z"], name="yyy"),
        )

        # 对df进行特定级别（"www"）的堆叠操作，结果存储在变量res中，并与更新后的exp进行比较验证
        res = df.stack("www")
        tm.assert_frame_equal(res, exp)
    # 测试子类化 DataFrame 的 stack 方法，混合使用多层索引
    def test_subclass_stack_multi_mixed(self):
        # GH 15564: GitHub 上的 issue 编号
        # 创建一个子类化的 DataFrame 实例 df，包含特定的数据和多层索引
        df = tm.SubclassedDataFrame(
            [
                [10, 11, 12.0, 13.0],
                [20, 21, 22.0, 23.0],
                [30, 31, 32.0, 33.0],
                [40, 41, 42.0, 43.0],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AABB"), list("cdcd"))), names=["aaa", "ccc"]
            ),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWXX"), list("yzyz"))), names=["www", "yyy"]
            ),
        )

        # 期望的结果 exp，也是一个子类化的 DataFrame 实例，包含特定的数据和多层索引
        exp = tm.SubclassedDataFrame(
            [
                [10, 12.0],
                [11, 13.0],
                [20, 22.0],
                [21, 23.0],
                [30, 32.0],
                [31, 33.0],
                [40, 42.0],
                [41, 43.0],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AAAABBBB"), list("ccddccdd"), list("yzyzyzyz"))),
                names=["aaa", "ccc", "yyy"],
            ),
            columns=Index(["W", "X"], name="www"),
        )

        # 对 df 调用 stack 方法，将 DataFrame 压缩至单层
        res = df.stack()
        # 使用测试工具 tm.assert_frame_equal 检查结果 res 与期望结果 exp 是否一致
        tm.assert_frame_equal(res, exp)

        # 对 df 调用 stack 方法，按照指定的列名 "yyy" 进行压缩
        res = df.stack("yyy")
        tm.assert_frame_equal(res, exp)

        # 更新期望的结果 exp，重新定义其中的数据和多层索引
        exp = tm.SubclassedDataFrame(
            [
                [10.0, 11.0],
                [12.0, 13.0],
                [20.0, 21.0],
                [22.0, 23.0],
                [30.0, 31.0],
                [32.0, 33.0],
                [40.0, 41.0],
                [42.0, 43.0],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AAAABBBB"), list("ccddccdd"), list("WXWXWXWX"))),
                names=["aaa", "ccc", "www"],
            ),
            columns=Index(["y", "z"], name="yyy"),
        )

        # 对 df 调用 stack 方法，按照指定的列名 "www" 进行压缩
        res = df.stack("www")
        tm.assert_frame_equal(res, exp)

    # 测试子类化 DataFrame 的 unstack 方法
    def test_subclass_unstack(self):
        # GH 15564: GitHub 上的 issue 编号
        # 创建一个子类化的 DataFrame 实例 df，包含特定的数据和简单索引
        df = tm.SubclassedDataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            index=["a", "b", "c"],
            columns=["X", "Y", "Z"],
        )

        # 对 df 调用 unstack 方法，展开 DataFrame 到多层 Series
        res = df.unstack()
        # 期望的结果 exp，是一个子类化的 Series 实例，包含特定的数据和多层索引
        exp = tm.SubclassedSeries(
            [1, 4, 7, 2, 5, 8, 3, 6, 9], index=[list("XXXYYYZZZ"), list("abcabcabc")]
        )

        # 使用测试工具 tm.assert_series_equal 检查结果 res 与期望结果 exp 是否一致
        tm.assert_series_equal(res, exp)
    def test_subclass_unstack_multi(self):
        # 测试子类化的DataFrame的多重展开功能
        # GH 15564
        # 创建一个子类化的DataFrame对象 df，包括数据和多级索引
        df = tm.SubclassedDataFrame(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]],
            index=MultiIndex.from_tuples(
                list(zip(list("AABB"), list("cdcd"))), names=["aaa", "ccc"]
            ),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWXX"), list("yzyz"))), names=["www", "yyy"]
            ),
        )

        # 预期的子类化DataFrame对象 exp，包括数据和多级索引
        exp = tm.SubclassedDataFrame(
            [[10, 20, 11, 21, 12, 22, 13, 23], [30, 40, 31, 41, 32, 42, 33, 43]],
            index=Index(["A", "B"], name="aaa"),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWWWXXXX"), list("yyzzyyzz"), list("cdcdcdcd"))),
                names=["www", "yyy", "ccc"],
            ),
        )

        # 对 df 进行 "unstack" 操作，得到结果 res，并断言其与 exp 相等
        res = df.unstack()
        tm.assert_frame_equal(res, exp)

        # 对 df 进行 "unstack" 操作，按照 "ccc" 进行展开，得到结果 res，并断言其与 exp 相等
        res = df.unstack("ccc")
        tm.assert_frame_equal(res, exp)

        # 更新预期的子类化DataFrame对象 exp，包括数据和多级索引
        exp = tm.SubclassedDataFrame(
            [[10, 30, 11, 31, 12, 32, 13, 33], [20, 40, 21, 41, 22, 42, 23, 43]],
            index=Index(["c", "d"], name="ccc"),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWWWXXXX"), list("yyzzyyzz"), list("ABABABAB"))),
                names=["www", "yyy", "aaa"],
            ),
        )

        # 对 df 进行 "unstack" 操作，按照 "aaa" 进行展开，得到结果 res，并断言其与 exp 相等
        res = df.unstack("aaa")
        tm.assert_frame_equal(res, exp)
    def test_subclass_unstack_multi_mixed(self):
        # GH 15564
        # 创建一个 SubclassedDataFrame 对象，包含多层索引和多层列，并初始化数据
        df = tm.SubclassedDataFrame(
            [
                [10, 11, 12.0, 13.0],
                [20, 21, 22.0, 23.0],
                [30, 31, 32.0, 33.0],
                [40, 41, 42.0, 43.0],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AABB"), list("cdcd"))), names=["aaa", "ccc"]
            ),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWXX"), list("yzyz"))), names=["www", "yyy"]
            ),
        )

        # 创建一个期望的 SubclassedDataFrame 对象，包含按预期组织的数据和索引/列结构
        exp = tm.SubclassedDataFrame(
            [
                [10, 20, 11, 21, 12.0, 22.0, 13.0, 23.0],
                [30, 40, 31, 41, 32.0, 42.0, 33.0, 43.0],
            ],
            index=Index(["A", "B"], name="aaa"),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWWWXXXX"), list("yyzzyyzz"), list("cdcdcdcd"))),
                names=["www", "yyy", "ccc"],
            ),
        )

        # 对 df 执行 unstack 操作，将多层索引的列转换为单层索引，与期望的结果比较
        res = df.unstack()
        tm.assert_frame_equal(res, exp)

        # 对 df 执行带参数 "ccc" 的 unstack 操作，将指定层次的列转换为行，与期望的结果比较
        res = df.unstack("ccc")
        tm.assert_frame_equal(res, exp)

        # 更新期望的 SubclassedDataFrame 对象，以适应另一种 unstack 操作的预期结果
        exp = tm.SubclassedDataFrame(
            [
                [10, 30, 11, 31, 12.0, 32.0, 13.0, 33.0],
                [20, 40, 21, 41, 22.0, 42.0, 23.0, 43.0],
            ],
            index=Index(["c", "d"], name="ccc"),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWWWXXXX"), list("yyzzyyzz"), list("ABABABAB"))),
                names=["www", "yyy", "aaa"],
            ),
        )

        # 对 df 执行带参数 "aaa" 的 unstack 操作，将指定层次的列转换为行，与更新后的期望结果比较
        res = df.unstack("aaa")
        tm.assert_frame_equal(res, exp)

    def test_subclass_pivot(self):
        # GH 15564
        # 创建一个 SubclassedDataFrame 对象，包含给定的数据和列名
        df = tm.SubclassedDataFrame(
            {
                "index": ["A", "B", "C", "C", "B", "A"],
                "columns": ["One", "One", "One", "Two", "Two", "Two"],
                "values": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            }
        )

        # 对 df 执行 pivot 操作，根据指定的索引和列转置数据
        pivoted = df.pivot(index="index", columns="columns", values="values")

        # 创建一个期望的 SubclassedDataFrame 对象，包含根据 pivot 操作的预期结果
        expected = tm.SubclassedDataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )

        # 设置期望结果的索引名和列名
        expected.index.name, expected.columns.name = "index", "columns"

        # 比较 pivot 后的结果与期望结果
        tm.assert_frame_equal(pivoted, expected)
    def test_subclassed_melt(self):
        # GH 15564
        # 创建一个自定义子类化的 DataFrame 对象 cheese，包含指定的列和数据
        cheese = tm.SubclassedDataFrame(
            {
                "first": ["John", "Mary"],
                "last": ["Doe", "Bo"],
                "height": [5.5, 6.0],
                "weight": [130, 150],
            }
        )

        # 对 cheese 进行融合操作，保留 'first' 和 'last' 列，其余列转换为 'variable' 和 'value'
        melted = pd.melt(cheese, id_vars=["first", "last"])

        # 创建一个预期的 DataFrame 对象 expected，包含指定的列和数据
        expected = tm.SubclassedDataFrame(
            [
                ["John", "Doe", "height", 5.5],
                ["Mary", "Bo", "height", 6.0],
                ["John", "Doe", "weight", 130],
                ["Mary", "Bo", "weight", 150],
            ],
            columns=["first", "last", "variable", "value"],
        )

        # 使用测试工具方法验证 melted 和 expected 的数据内容是否相同
        tm.assert_frame_equal(melted, expected)

    def test_subclassed_wide_to_long(self):
        # GH 9762
        # 创建一个包含指定列和数据的自定义子类化的 DataFrame 对象 df
        x = np.random.default_rng(2).standard_normal(3)
        df = tm.SubclassedDataFrame(
            {
                "A1970": {0: "a", 1: "b", 2: "c"},
                "A1980": {0: "d", 1: "e", 2: "f"},
                "B1970": {0: 2.5, 1: 1.2, 2: 0.7},
                "B1980": {0: 3.2, 1: 1.3, 2: 0.1},
                "X": dict(zip(range(3), x)),
            }
        )

        # 在 df 上新增一列 'id'，其值为索引值
        df["id"] = df.index

        # 创建预期的数据字典 exp_data，包含指定列和数据
        exp_data = {
            "X": x.tolist() + x.tolist(),
            "A": ["a", "b", "c", "d", "e", "f"],
            "B": [2.5, 1.2, 0.7, 3.2, 1.3, 0.1],
            "year": [1970, 1970, 1970, 1980, 1980, 1980],
            "id": [0, 1, 2, 0, 1, 2],
        }

        # 创建预期的 DataFrame 对象 expected，设置索引为 ('id', 'year')，并选择列 ['X', 'A', 'B']
        expected = tm.SubclassedDataFrame(exp_data)
        expected = expected.set_index(["id", "year"])[["X", "A", "B"]]

        # 使用 pd.wide_to_long 函数将 df 转换为长格式 DataFrame，根据 'id' 和 'year' 进行重塑
        long_frame = pd.wide_to_long(df, ["A", "B"], i="id", j="year")

        # 使用测试工具方法验证 long_frame 和 expected 的数据内容是否相同
        tm.assert_frame_equal(long_frame, expected)
    # 定义一个测试方法，用于测试子类化 DataFrame 的 apply 方法
    def test_subclassed_apply(self):
        # GH 19822
        # 定义一个内部函数，用于检查行是否为 SubclassedSeries 的实例
        def check_row_subclass(row):
            assert isinstance(row, tm.SubclassedSeries)

        # 定义一个内部函数，用于处理行数据，根据条件修改 "height" 变量的值
        def stretch(row):
            if row["variable"] == "height":
                row["value"] += 0.5
            return row

        # 创建一个 SubclassedDataFrame 对象，包含一些测试数据
        df = tm.SubclassedDataFrame(
            [
                ["John", "Doe", "height", 5.5],
                ["Mary", "Bo", "height", 6.0],
                ["John", "Doe", "weight", 130],
                ["Mary", "Bo", "weight", 150],
            ],
            columns=["first", "last", "variable", "value"],
        )

        # 使用 apply 方法调用 check_row_subclass 函数，检查每列是否为 SubclassedSeries 实例
        df.apply(lambda x: check_row_subclass(x))
        # 使用 apply 方法调用 check_row_subclass 函数，检查每行是否为 SubclassedSeries 实例
        df.apply(lambda x: check_row_subclass(x), axis=1)

        # 创建一个期望的 SubclassedDataFrame 对象，用于后续的比较
        expected = tm.SubclassedDataFrame(
            [
                ["John", "Doe", "height", 6.0],
                ["Mary", "Bo", "height", 6.5],
                ["John", "Doe", "weight", 130],
                ["Mary", "Bo", "weight", 150],
            ],
            columns=["first", "last", "variable", "value"],
        )

        # 使用 apply 方法调用 stretch 函数，按行处理数据，修改 "height" 变量的值
        result = df.apply(lambda x: stretch(x), axis=1)
        # 断言结果是否为 SubclassedDataFrame 的实例
        assert isinstance(result, tm.SubclassedDataFrame)
        # 断言处理后的结果与期望的结果是否相等
        tm.assert_frame_equal(result, expected)

        # 创建一个期望的 SubclassedDataFrame 对象，其中包含固定的列表作为每行的返回结果
        expected = tm.SubclassedDataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])

        # 使用 apply 方法调用一个返回 SubclassedSeries 对象的 lambda 函数，按行处理数据
        result = df.apply(lambda x: tm.SubclassedSeries([1, 2, 3]), axis=1)
        # 断言结果是否为 SubclassedDataFrame 的实例
        assert isinstance(result, tm.SubclassedDataFrame)
        # 断言处理后的结果与期望的结果是否相等
        tm.assert_frame_equal(result, expected)

        # 使用 apply 方法调用一个返回列表的 lambda 函数，按行处理数据，指定 result_type 为 "expand"
        result = df.apply(lambda x: [1, 2, 3], axis=1, result_type="expand")
        # 断言结果是否为 SubclassedDataFrame 的实例
        assert isinstance(result, tm.SubclassedDataFrame)
        # 断言处理后的结果与期望的结果是否相等
        tm.assert_frame_equal(result, expected)

        # 创建一个期望的 SubclassedSeries 对象，其中包含固定的列表作为每行的返回结果
        expected = tm.SubclassedSeries([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])

        # 使用 apply 方法调用一个返回列表的 lambda 函数，按行处理数据
        result = df.apply(lambda x: [1, 2, 3], axis=1)
        # 断言结果不是 SubclassedDataFrame 的实例
        assert not isinstance(result, tm.SubclassedDataFrame)
        # 断言处理后的结果与期望的结果是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，用于测试子类化 DataFrame 的 reductions
    def test_subclassed_reductions(self, all_reductions):
        # GH 25596

        # 创建一个 SubclassedDataFrame 对象，包含一个字典作为数据
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        # 调用 getattr 函数获取指定 reductions 方法的结果
        result = getattr(df, all_reductions)()
        # 断言结果是否为 SubclassedSeries 的实例
        assert isinstance(result, tm.SubclassedSeries)
    # 测试自定义子类化的 DataFrame 的 count 方法
    def test_subclassed_count(self):
        # 创建一个 SubclassedDataFrame 对象，包含不同类型的数据列
        df = tm.SubclassedDataFrame(
            {
                "Person": ["John", "Myla", "Lewis", "John", "Myla"],
                "Age": [24.0, np.nan, 21.0, 33, 26],
                "Single": [False, True, True, True, False],
            }
        )
        # 调用 count 方法，返回结果
        result = df.count()
        # 断言结果类型为 SubclassedSeries
        assert isinstance(result, tm.SubclassedSeries)

        # 创建另一个 SubclassedDataFrame 对象，包含不同的数据列
        df = tm.SubclassedDataFrame({"A": [1, 0, 3], "B": [0, 5, 6], "C": [7, 8, 0]})
        # 调用 count 方法，返回结果
        result = df.count()
        # 断言结果类型为 SubclassedSeries
        assert isinstance(result, tm.SubclassedSeries)

        # 创建具有多级索引和列的 SubclassedDataFrame 对象
        df = tm.SubclassedDataFrame(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]],
            index=MultiIndex.from_tuples(
                list(zip(list("AABB"), list("cdcd"))), names=["aaa", "ccc"]
            ),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWXX"), list("yzyz"))), names=["www", "yyy"]
            ),
        )
        # 调用 count 方法，返回结果
        result = df.count()
        # 断言结果类型为 SubclassedSeries
        assert isinstance(result, tm.SubclassedSeries)

        # 创建一个空的 SubclassedDataFrame 对象
        df = tm.SubclassedDataFrame()
        # 调用 count 方法，返回结果
        result = df.count()
        # 断言结果类型为 SubclassedSeries
        assert isinstance(result, tm.SubclassedSeries)

    # 测试自定义子类化的 DataFrame 的 isin 方法
    def test_isin(self):
        # 创建一个 SubclassedDataFrame 对象，包含数字列和索引
        df = tm.SubclassedDataFrame(
            {"num_legs": [2, 4], "num_wings": [2, 0]}, index=["falcon", "dog"]
        )
        # 调用 isin 方法，返回结果
        result = df.isin([0, 2])
        # 断言结果类型为 SubclassedDataFrame
        assert isinstance(result, tm.SubclassedDataFrame)

    # 测试自定义子类化的 DataFrame 的 duplicated 方法
    def test_duplicated(self):
        # 创建一个 SubclassedDataFrame 对象，包含多个数字列
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        # 调用 duplicated 方法，返回结果
        result = df.duplicated()
        # 断言结果类型为 SubclassedSeries
        assert isinstance(result, tm.SubclassedSeries)

        # 创建一个空的 SubclassedDataFrame 对象
        df = tm.SubclassedDataFrame()
        # 调用 duplicated 方法，返回结果
        result = df.duplicated()
        # 断言结果类型为 SubclassedSeries
        assert isinstance(result, tm.SubclassedSeries)

    # 使用 pytest 的参数化功能，测试自定义子类化的 DataFrame 的 idxmax 和 idxmin 方法
    @pytest.mark.parametrize("idx_method", ["idxmax", "idxmin"])
    def test_idx(self, idx_method):
        # 创建一个 SubclassedDataFrame 对象，包含多个数字列
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        # 使用 getattr 调用指定的 idx_method 方法，返回结果
        result = getattr(df, idx_method)()
        # 断言结果类型为 SubclassedSeries
        assert isinstance(result, tm.SubclassedSeries)

    # 测试自定义子类化的 DataFrame 的 dot 方法
    def test_dot(self):
        # 创建一个 SubclassedDataFrame 对象，包含多个数字列
        df = tm.SubclassedDataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
        # 创建一个 SubclassedSeries 对象
        s = tm.SubclassedSeries([1, 1, 2, 1])
        # 调用 dot 方法，返回结果
        result = df.dot(s)
        # 断言结果类型为 SubclassedSeries
        assert isinstance(result, tm.SubclassedSeries)

        # 创建一个 SubclassedDataFrame 对象，包含多个数字列
        df = tm.SubclassedDataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
        # 创建一个不同结构的 SubclassedDataFrame 对象
        s = tm.SubclassedDataFrame([1, 1, 2, 1])
        # 调用 dot 方法，返回结果
        result = df.dot(s)
        # 断言结果类型为 SubclassedDataFrame
        assert isinstance(result, tm.SubclassedDataFrame)

    # 测试自定义子类化的 DataFrame 的 memory_usage 方法
    def test_memory_usage(self):
        # 创建一个 SubclassedDataFrame 对象，包含多个数字列
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        # 调用 memory_usage 方法，返回结果
        result = df.memory_usage()
        # 断言结果类型为 SubclassedSeries
        assert isinstance(result, tm.SubclassedSeries)

        # 调用 memory_usage 方法，使用 index=False 参数，返回结果
        result = df.memory_usage(index=False)
        # 断言结果类型为 SubclassedSeries
        assert isinstance(result, tm.SubclassedSeries)
    def test_corrwith(self):
        # 确保导入 scipy 库成功，如果失败则跳过该测试
        pytest.importorskip("scipy")
        
        # 定义 DataFrame 的行索引和列标签
        index = ["a", "b", "c", "d", "e"]
        columns = ["one", "two", "three", "four"]
        
        # 创建第一个子类化的 DataFrame，使用正态分布生成随机数据
        df1 = tm.SubclassedDataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            index=index,
            columns=columns,
        )
        
        # 创建第二个子类化的 DataFrame，使用相同的随机种子和维度生成数据
        df2 = tm.SubclassedDataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=index[:4],
            columns=columns,
        )
        
        # 计算两个 DataFrame 之间的相关性，使用 Kendall 方法，按行计算并且忽略缺失值
        correls = df1.corrwith(df2, axis=1, drop=True, method="kendall")

        # 断言计算结果为子类化的 Series 对象
        assert isinstance(correls, (tm.SubclassedSeries))

    def test_asof(self):
        # 定义时间序列长度
        N = 3
        
        # 生成日期时间索引，频率为 53 秒
        rng = pd.date_range("1/1/1990", periods=N, freq="53s")
        
        # 创建子类化的 DataFrame，所有值初始化为 NaN，使用时间索引
        df = tm.SubclassedDataFrame(
            {
                "A": [np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan],
                "C": [np.nan, np.nan, np.nan],
            },
            index=rng,
        )

        # 使用 asof 方法查询最后两个时间点的数据，并断言返回结果为子类化的 DataFrame
        result = df.asof(rng[-2:])
        assert isinstance(result, tm.SubclassedDataFrame)

        # 使用 asof 方法查询最后一个时间点的数据，并断言返回结果为子类化的 Series
        result = df.asof(rng[-2])
        assert isinstance(result, tm.SubclassedSeries)

        # 使用 asof 方法查询指定日期之前最近的数据，并断言返回结果为子类化的 Series
        result = df.asof("1989-12-31")
        assert isinstance(result, tm.SubclassedSeries)

    def test_idxmin_preserves_subclass(self):
        # GH 28330

        # 创建子类化的 DataFrame，包含三列数据
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        
        # 找出每列最小值的索引，断言返回结果为子类化的 Series
        result = df.idxmin()
        assert isinstance(result, tm.SubclassedSeries)

    def test_idxmax_preserves_subclass(self):
        # GH 28330

        # 创建子类化的 DataFrame，包含三列数据
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        
        # 找出每列最大值的索引，断言返回结果为子类化的 Series
        result = df.idxmax()
        assert isinstance(result, tm.SubclassedSeries)

    def test_convert_dtypes_preserves_subclass(self):
        # GH 43668
        
        # 创建子类化的 DataFrame，包含三列数据
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        
        # 执行 convert_dtypes 方法，将列数据类型转换为最适合的类型，断言返回结果为子类化的 DataFrame
        result = df.convert_dtypes()
        assert isinstance(result, tm.SubclassedDataFrame)

    def test_convert_dtypes_preserves_subclass_with_constructor(self):
        # 使用自定义构造器的子类化 DataFrame
        
        # 定义一个新的 DataFrame 类，继承自 pandas 的 DataFrame
        class SubclassedDataFrame(DataFrame):
            @property
            def _constructor(self):
                return SubclassedDataFrame
        
        # 创建一个实例化的子类化 DataFrame，包含一列数据
        df = SubclassedDataFrame({"a": [1, 2, 3]})
        
        # 执行 convert_dtypes 方法，将列数据类型转换为最适合的类型，断言返回结果为自定义构造器的子类化 DataFrame
        result = df.convert_dtypes()
        assert isinstance(result, SubclassedDataFrame)

    def test_astype_preserves_subclass(self):
        # GH#40810

        # 创建子类化的 DataFrame，包含三列数据
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        
        # 执行 astype 方法，将指定列数据类型转换为相应类型，断言返回结果为子类化的 DataFrame
        result = df.astype({"A": np.int64, "B": np.int32, "C": np.float64})
        assert isinstance(result, tm.SubclassedDataFrame)

    def test_equals_subclass(self):
        # https://github.com/pandas-dev/pandas/pull/34402
        # 允许在两个方向上使用子类化

        # 创建普通的 DataFrame 和子类化的 DataFrame，包含相同的数据
        df1 = DataFrame({"a": [1, 2, 3]})
        df2 = tm.SubclassedDataFrame({"a": [1, 2, 3]})
        
        # 断言两个 DataFrame 相等，从而验证 equals 方法对子类化的支持
        assert df1.equals(df2)
        assert df2.equals(df1)
class MySubclassWithMetadata(DataFrame):
    _metadata = ["my_metadata"]  # 定义类级别的元数据列表，用于存储自定义元数据信息

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # 调用父类的构造函数初始化实例

        my_metadata = kwargs.pop("my_metadata", None)  # 从关键字参数中获取并移除 my_metadata
        if args and isinstance(args[0], MySubclassWithMetadata):
            my_metadata = args[0].my_metadata  # 如果第一个参数是 MySubclassWithMetadata 的实例，则继承其元数据
        self.my_metadata = my_metadata  # 将元数据存储在实例属性中

    @property
    def _constructor(self):
        return MySubclassWithMetadata  # 返回用于构造新实例的类，即 MySubclassWithMetadata


def test_constructor_with_metadata():
    # 测试构造函数能正确处理元数据的情况
    # 参考链接：https://github.com/pandas-dev/pandas/pull/54922
    # 参考链接：https://github.com/pandas-dev/pandas/issues/55120
    df = MySubclassWithMetadata(
        np.random.default_rng(2).random((5, 3)), columns=["A", "B", "C"]
    )  # 创建 MySubclassWithMetadata 的实例，传入随机数据和列名
    subset = df[["A", "B"]]  # 获取数据框的子集
    assert isinstance(subset, MySubclassWithMetadata)  # 断言子集也是 MySubclassWithMetadata 的实例


class SimpleDataFrameSubClass(DataFrame):
    """一个未定义构造函数的 DataFrame 的子类。"""


class SimpleSeriesSubClass(Series):
    """一个未定义构造函数的 Series 的子类。"""


class TestSubclassWithoutConstructor:
    def test_copy_df(self):
        expected = DataFrame({"a": [1, 2, 3]})
        result = SimpleDataFrameSubClass(expected).copy()  # 复制 SimpleDataFrameSubClass 的实例

        assert (
            type(result) is DataFrame
        )  # 使用 type() 断言结果是 DataFrame 类型
        tm.assert_frame_equal(result, expected)  # 使用测试工具函数比较结果和期望的 DataFrame

    def test_copy_series(self):
        expected = Series([1, 2, 3])
        result = SimpleSeriesSubClass(expected).copy()  # 复制 SimpleSeriesSubClass 的实例

        tm.assert_series_equal(result, expected)  # 使用测试工具函数比较结果和期望的 Series

    def test_series_to_frame(self):
        orig = Series([1, 2, 3])
        expected = orig.to_frame()  # 将原始 Series 转换为 DataFrame 作为期望结果
        result = SimpleSeriesSubClass(orig).to_frame()  # 将 SimpleSeriesSubClass 的实例转换为 DataFrame

        assert (
            type(result) is DataFrame
        )  # 使用 type() 断言结果是 DataFrame 类型
        tm.assert_frame_equal(result, expected)  # 使用测试工具函数比较结果和期望的 DataFrame

    def test_groupby(self):
        df = SimpleDataFrameSubClass(DataFrame({"a": [1, 2, 3]}))

        for _, v in df.groupby("a"):
            assert type(v) is DataFrame  # 遍历分组后的数据框，断言每个分组是 DataFrame 类型
```