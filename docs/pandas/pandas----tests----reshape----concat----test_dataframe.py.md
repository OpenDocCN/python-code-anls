# `D:\src\scipysrc\pandas\pandas\tests\reshape\concat\test_dataframe.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 pytest 库，用于单元测试

import pandas as pd  # 导入 Pandas 库，用于数据分析
from pandas import (  # 从 Pandas 中导入特定模块和函数
    DataFrame,       # 数据框
    Index,           # 索引
    Series,          # 系列
    concat,          # 连接函数
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具模块

class TestDataFrameConcat:
    def test_concat_multiple_frames_dtypes(self):
        # GH#2759
        # 创建包含浮点数数据的 DataFrame df1 和 df2
        df1 = DataFrame(data=np.ones((10, 2)), columns=["foo", "bar"], dtype=np.float64)
        df2 = DataFrame(data=np.ones((10, 2)), dtype=np.float32)
        # 进行沿着列轴的连接，并获取结果的数据类型
        results = concat((df1, df2), axis=1).dtypes
        # 创建预期的数据类型 Series
        expected = Series(
            [np.dtype("float64")] * 2 + [np.dtype("float32")] * 2,
            index=["foo", "bar", 0, 1],
        )
        tm.assert_series_equal(results, expected)

    def test_concat_tuple_keys(self):
        # GH#14438
        # 创建包含数据的 DataFrame df1 和 df2
        df1 = DataFrame(np.ones((2, 2)), columns=list("AB"))
        df2 = DataFrame(np.ones((3, 2)) * 2, columns=list("AB"))
        # 进行带有元组键的连接操作
        results = concat((df1, df2), keys=[("bee", "bah"), ("bee", "boo")])
        # 创建预期的 DataFrame 结果
        expected = DataFrame(
            {
                "A": {
                    ("bee", "bah", 0): 1.0,
                    ("bee", "bah", 1): 1.0,
                    ("bee", "boo", 0): 2.0,
                    ("bee", "boo", 1): 2.0,
                    ("bee", "boo", 2): 2.0,
                },
                "B": {
                    ("bee", "bah", 0): 1.0,
                    ("bee", "bah", 1): 1.0,
                    ("bee", "boo", 0): 2.0,
                    ("bee", "boo", 1): 2.0,
                    ("bee", "boo", 2): 2.0,
                },
            }
        )
        tm.assert_frame_equal(results, expected)

    def test_concat_named_keys(self):
        # GH#14252
        # 创建包含具名列的 DataFrame df
        df = DataFrame({"foo": [1, 2], "bar": [0.1, 0.2]})
        index = Index(["a", "b"], name="baz")
        # 使用具名键进行连接
        concatted_named_from_keys = concat([df, df], keys=index)
        # 创建预期的具名索引 DataFrame 结果
        expected_named = DataFrame(
            {"foo": [1, 2, 1, 2], "bar": [0.1, 0.2, 0.1, 0.2]},
            index=pd.MultiIndex.from_product((["a", "b"], [0, 1]), names=["baz", None]),
        )
        tm.assert_frame_equal(concatted_named_from_keys, expected_named)

        index_no_name = Index(["a", "b"], name=None)
        # 使用指定的名称进行连接
        concatted_named_from_names = concat([df, df], keys=index_no_name, names=["baz"])
        tm.assert_frame_equal(concatted_named_from_names, expected_named)

        # 进行未命名的连接
        concatted_unnamed = concat([df, df], keys=index_no_name)
        # 创建预期的未命名索引 DataFrame 结果
        expected_unnamed = DataFrame(
            {"foo": [1, 2, 1, 2], "bar": [0.1, 0.2, 0.1, 0.2]},
            index=pd.MultiIndex.from_product((["a", "b"], [0, 1]), names=[None, None]),
        )
        tm.assert_frame_equal(concatted_unnamed, expected_unnamed)
    def test_concat_axis_parameter(self):
        # GH#14369
        # 创建包含两列A的DataFrame，每列有两个元素
        df1 = DataFrame({"A": [0.1, 0.2]}, index=range(2))
        df2 = DataFrame({"A": [0.3, 0.4]}, index=range(2))

        # 创建期望的索引/行/0 DataFrame
        expected_index = DataFrame({"A": [0.1, 0.2, 0.3, 0.4]}, index=[0, 1, 0, 1])

        # 沿着索引轴拼接DataFrame
        concatted_index = concat([df1, df2], axis="index")
        tm.assert_frame_equal(concatted_index, expected_index)

        # 沿着行轴拼接DataFrame
        concatted_row = concat([df1, df2], axis="rows")
        tm.assert_frame_equal(concatted_row, expected_index)

        # 与 axis=0 等效的行轴拼接DataFrame
        concatted_0 = concat([df1, df2], axis=0)
        tm.assert_frame_equal(concatted_0, expected_index)

        # 创建期望的列/1 DataFrame
        expected_columns = DataFrame(
            [[0.1, 0.3], [0.2, 0.4]], index=[0, 1], columns=["A", "A"]
        )

        # 沿着列轴拼接DataFrame
        concatted_columns = concat([df1, df2], axis="columns")
        tm.assert_frame_equal(concatted_columns, expected_columns)

        # 与 axis=1 等效的列轴拼接DataFrame
        concatted_1 = concat([df1, df2], axis=1)
        tm.assert_frame_equal(concatted_1, expected_columns)

        # 创建两个Series
        series1 = Series([0.1, 0.2])
        series2 = Series([0.3, 0.4])

        # 创建期望的索引/行/0 Series
        expected_index_series = Series([0.1, 0.2, 0.3, 0.4], index=[0, 1, 0, 1])

        # 沿着索引轴拼接Series
        concatted_index_series = concat([series1, series2], axis="index")
        tm.assert_series_equal(concatted_index_series, expected_index_series)

        # 沿着行轴拼接Series
        concatted_row_series = concat([series1, series2], axis="rows")
        tm.assert_series_equal(concatted_row_series, expected_index_series)

        # 与 axis=0 等效的行轴拼接Series
        concatted_0_series = concat([series1, series2], axis=0)
        tm.assert_series_equal(concatted_0_series, expected_index_series)

        # 创建期望的列/1 Series
        expected_columns_series = DataFrame(
            [[0.1, 0.3], [0.2, 0.4]], index=[0, 1], columns=[0, 1]
        )

        # 沿着列轴拼接Series
        concatted_columns_series = concat([series1, series2], axis="columns")
        tm.assert_frame_equal(concatted_columns_series, expected_columns_series)

        # 与 axis=1 等效的列轴拼接Series
        concatted_1_series = concat([series1, series2], axis=1)
        tm.assert_frame_equal(concatted_1_series, expected_columns_series)

        # 测试 ValueError 异常
        with pytest.raises(ValueError, match="No axis named"):
            # 拼接时指定错误的轴参数
            concat([series1, series2], axis="something")
    def test_concat_astype_dup_col(self):
        # GH#23049
        # 创建包含单个字典的DataFrame
        df = DataFrame([{"a": "b"}])
        # 沿着列轴连接两次df，形成新的DataFrame
        df = concat([df, df], axis=1)

        # 将DataFrame转换为category类型
        result = df.astype("category")
        # 创建预期的DataFrame，包含两次重复的值"b"，并转换为category类型
        expected = DataFrame(
            np.array(["b", "b"]).reshape(1, 2), columns=["a", "a"]
        ).astype("category")
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    def test_concat_dataframe_keys_bug(self, sort):
        # 创建包含单个Series的DataFrame t1
        t1 = DataFrame(
            {"value": Series([1, 2, 3], index=Index(["a", "b", "c"], name="id"))}
        )
        # 创建包含单个Series的DataFrame t2
        t2 = DataFrame({"value": Series([7, 8], index=Index(["a", "b"], name="id"))})

        # 沿着列轴连接t1和t2，并分别标记为"t1"和"t2"
        result = concat([t1, t2], axis=1, keys=["t1", "t2"], sort=sort)
        # 断言结果DataFrame的列标签是否为 [("t1", "value"), ("t2", "value")]
        assert list(result.columns) == [("t1", "value"), ("t2", "value")]

    def test_concat_bool_with_int(self):
        # GH#42092 可能需要更改返回类型为object，但需要废弃当前的用法
        # 创建包含单个bool类型Series的DataFrame df1
        df1 = DataFrame(Series([True, False, True, True], dtype="bool"))
        # 创建包含单个int64类型Series的DataFrame df2
        df2 = DataFrame(Series([1, 0, 1], dtype="int64"))

        # 沿着行轴连接df1和df2
        result = concat([df1, df2])
        # 创建预期的DataFrame，将df1转换为int64类型后与df2连接
        expected = concat([df1.astype("int64"), df2])
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    def test_concat_duplicates_in_index_with_keys(self):
        # GH#42651
        # 创建包含重复索引的DataFrame
        index = [1, 1, 3]
        data = [1, 2, 3]

        df = DataFrame(data=data, index=index)
        # 沿着行轴连接df，并使用"ID"和"date"作为键和名称
        result = concat([df], keys=["A"], names=["ID", "date"])
        # 创建预期的DataFrame，使用MultiIndex作为索引
        mi = pd.MultiIndex.from_product([["A"], index], names=["ID", "date"])
        expected = DataFrame(data=data, index=mi)
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)
        # 断言结果DataFrame的第二级索引是否与预期相等
        tm.assert_index_equal(result.index.levels[1], Index([1, 3], name="date"))

    def test_outer_sort_columns(self):
        # GH#47127
        # 创建包含特定列名和数据的DataFrame df1
        df1 = DataFrame({"A": [0], "B": [1], 0: 1})
        # 创建包含特定列名和数据的DataFrame df2
        df2 = DataFrame({"A": [100]})
        # 沿着行轴连接df1和df2，使用外连接方式并忽略索引，按列名排序
        result = concat([df1, df2], ignore_index=True, join="outer", sort=True)
        # 创建预期的DataFrame，包含合并后的列，并按列名排序
        expected = DataFrame({0: [1.0, np.nan], "A": [0, 100], "B": [1.0, np.nan]})
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    def test_inner_sort_columns(self):
        # GH#47127
        # 创建包含特定列名和数据的DataFrame df1
        df1 = DataFrame({"A": [0], "B": [1], 0: 1})
        # 创建包含特定列名和数据的DataFrame df2
        df2 = DataFrame({"A": [100], 0: 2})
        # 沿着行轴连接df1和df2，使用内连接方式并忽略索引，按列名排序
        result = concat([df1, df2], ignore_index=True, join="inner", sort=True)
        # 创建预期的DataFrame，包含合并后的列，并按列名排序
        expected = DataFrame({0: [1, 2], "A": [0, 100]})
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    def test_sort_columns_one_df(self):
        # GH#47127
        # 创建包含特定列名和数据的DataFrame df1
        df1 = DataFrame({"A": [100], 0: 2})
        # 沿着行轴连接df1，使用内连接方式并忽略索引，按列名排序
        result = concat([df1], ignore_index=True, join="inner", sort=True)
        # 创建预期的DataFrame，包含合并后的列，并按列名排序
        expected = DataFrame({0: [2], "A": [100]})
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)
```