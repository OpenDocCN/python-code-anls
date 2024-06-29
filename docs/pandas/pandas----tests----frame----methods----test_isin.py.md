# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_isin.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值数据
import pytest  # 导入 pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，用于数据分析和处理
from pandas import (  # 从 Pandas 中导入 DataFrame、MultiIndex、Series 类
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块，用于测试辅助

class TestDataFrameIsIn:
    def test_isin(self):
        # GH#4211
        df = DataFrame(  # 创建一个 DataFrame 对象
            {
                "vals": [1, 2, 3, 4],
                "ids": ["a", "b", "f", "n"],
                "ids2": ["a", "n", "c", "n"],
            },
            index=["foo", "bar", "baz", "qux"],  # 设置索引
        )
        other = ["a", "b", "c"]

        result = df.isin(other)  # 对 DataFrame 应用 isin 方法，返回匹配结果
        expected = DataFrame([df.loc[s].isin(other) for s in df.index])  # 期望的匹配结果 DataFrame
        tm.assert_frame_equal(result, expected)  # 使用测试辅助函数检查结果是否相等

    @pytest.mark.parametrize("empty", [[], Series(dtype=object), np.array([])])
    def test_isin_empty(self, empty):
        # GH#16991
        df = DataFrame({"A": ["a", "b", "c"], "B": ["a", "e", "f"]})  # 创建一个新的 DataFrame
        expected = DataFrame(False, df.index, df.columns)  # 期望的结果 DataFrame，初始化为 False

        result = df.isin(empty)  # 对 DataFrame 应用 isin 方法，返回匹配结果
        tm.assert_frame_equal(result, expected)  # 使用测试辅助函数检查结果是否相等

    def test_isin_dict(self):
        df = DataFrame({"A": ["a", "b", "c"], "B": ["a", "e", "f"]})  # 创建一个新的 DataFrame
        d = {"A": ["a"]}  # 定义一个字典 d

        expected = DataFrame(False, df.index, df.columns)  # 期望的结果 DataFrame，初始化为 False
        expected.loc[0, "A"] = True  # 在期望的结果 DataFrame 中设置匹配值为 True

        result = df.isin(d)  # 对 DataFrame 应用 isin 方法，返回匹配结果
        tm.assert_frame_equal(result, expected)  # 使用测试辅助函数检查结果是否相等

        # non unique columns
        df = DataFrame({"A": ["a", "b", "c"], "B": ["a", "e", "f"]})  # 创建一个新的 DataFrame，列名重复
        df.columns = ["A", "A"]  # 修改列名为非唯一的情况
        expected = DataFrame(False, df.index, df.columns)  # 期望的结果 DataFrame，初始化为 False
        expected.loc[0, "A"] = True  # 在期望的结果 DataFrame 中设置匹配值为 True
        result = df.isin(d)  # 对 DataFrame 应用 isin 方法，返回匹配结果
        tm.assert_frame_equal(result, expected)  # 使用测试辅助函数检查结果是否相等

    def test_isin_with_string_scalar(self):
        # GH#4763
        df = DataFrame(  # 创建一个 DataFrame 对象
            {
                "vals": [1, 2, 3, 4],
                "ids": ["a", "b", "f", "n"],
                "ids2": ["a", "n", "c", "n"],
            },
            index=["foo", "bar", "baz", "qux"],  # 设置索引
        )
        msg = (
            r"only list-like or dict-like objects are allowed "  # 错误消息字符串
            r"to be passed to DataFrame.isin\(\), you passed a 'str'"  # 错误消息字符串的一部分
        )
        with pytest.raises(TypeError, match=msg):  # 检查是否引发了预期的 TypeError 异常
            df.isin("a")

        with pytest.raises(TypeError, match=msg):  # 检查是否引发了预期的 TypeError 异常
            df.isin("aaa")

    def test_isin_df(self):
        df1 = DataFrame({"A": [1, 2, 3, 4], "B": [2, np.nan, 4, 4]})  # 创建第一个 DataFrame
        df2 = DataFrame({"A": [0, 2, 12, 4], "B": [2, np.nan, 4, 5]})  # 创建第二个 DataFrame
        expected = DataFrame(False, df1.index, df1.columns)  # 期望的结果 DataFrame，初始化为 False
        result = df1.isin(df2)  # 对第一个 DataFrame 应用 isin 方法，传入第二个 DataFrame

        expected.loc[[1, 3], "A"] = True  # 在期望的结果 DataFrame 中设置匹配值为 True
        expected.loc[[0, 2], "B"] = True  # 在期望的结果 DataFrame 中设置匹配值为 True

        tm.assert_frame_equal(result, expected)  # 使用测试辅助函数检查结果是否相等

        # partial overlapping columns
        df2.columns = ["A", "C"]  # 修改第二个 DataFrame 的列名，部分重叠
        result = df1.isin(df2)  # 对第一个 DataFrame 应用 isin 方法，传入修改后的第二个 DataFrame

        expected["B"] = False  # 重置部分列匹配结果为 False
        tm.assert_frame_equal(result, expected)  # 使用测试辅助函数检查结果是否相等
    # 测试 DataFrame 的 isin 方法，验证 GH#16394 的问题修复
    def test_isin_tuples(self):
        # 创建一个包含两列的 DataFrame，其中一列是整数，另一列是字符串
        df = DataFrame({"A": [1, 2, 3], "B": ["a", "b", "f"]})
        # 将两列合并成元组并赋给新列 "C"
        df["C"] = list(zip(df["A"], df["B"]))
        # 使用 isin 方法检查元组列表 [(1, "a")] 是否存在于 "C" 列中
        result = df["C"].isin([(1, "a")])
        # 使用断言验证结果是否与预期的 Series 相等
        tm.assert_series_equal(result, Series([True, False, False], name="C"))

    # 测试 DataFrame 的 isin 方法，验证处理含有重复值的 DataFrame 的情况
    def test_isin_df_dupe_values(self):
        # 创建第一个 DataFrame，包含两列，其中一列包含 NaN 值
        df1 = DataFrame({"A": [1, 2, 3, 4], "B": [2, np.nan, 4, 4]})
        # 创建第二个 DataFrame，只有列名重复
        df2 = DataFrame([[0, 2], [12, 4], [2, np.nan], [4, 5]], columns=["B", "B"])
        # 准备错误消息的正则表达式模式
        msg = r"cannot compute isin with a duplicate axis\."
        # 使用 pytest 的断言检查是否引发了 ValueError，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            df1.isin(df2)

        # 创建第二个 DataFrame，只有索引重复
        df2 = DataFrame(
            [[0, 2], [12, 4], [2, np.nan], [4, 5]],
            columns=["A", "B"],
            index=[0, 0, 1, 1],
        )
        # 使用 pytest 的断言检查是否引发了 ValueError，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            df1.isin(df2)

        # 创建第二个 DataFrame，同时列名和索引都重复
        df2.columns = ["B", "B"]
        # 使用 pytest 的断言检查是否引发了 ValueError，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            df1.isin(df2)

    # 测试 DataFrame 的 isin 方法，验证处理含有自身列重复值的情况
    def test_isin_dupe_self(self):
        # 创建另一个 DataFrame，含有重复值的情况
        other = DataFrame({"A": [1, 0, 1, 0], "B": [1, 1, 0, 0]})
        # 创建一个含有重复列的 DataFrame
        df = DataFrame([[1, 1], [1, 0], [0, 0]], columns=["A", "A"])
        # 使用 isin 方法检查 df 是否包含在 other 中
        result = df.isin(other)
        # 创建一个预期的 DataFrame，其中与 other 匹配的位置为 True，其余为 False
        expected = DataFrame(False, index=df.index, columns=df.columns)
        expected.loc[0] = True
        expected.iloc[1, 1] = True
        # 使用断言验证结果是否与预期的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    # 测试 DataFrame 的 isin 方法，验证处理与 Series 比较的情况
    def test_isin_against_series(self):
        # 创建一个含有索引的 DataFrame 和一个 Series
        df = DataFrame(
            {"A": [1, 2, 3, 4], "B": [2, np.nan, 4, 4]}, index=["a", "b", "c", "d"]
        )
        s = Series([1, 3, 11, 4], index=["a", "b", "c", "d"])
        # 创建一个预期的 DataFrame，其中与 Series 匹配的位置为 True，其余为 False
        expected = DataFrame(False, index=df.index, columns=df.columns)
        expected.loc["a", "A"] = True
        expected.loc["d"] = True
        # 使用 isin 方法检查 df 是否包含在 s 中
        result = df.isin(s)
        # 使用断言验证结果是否与预期的 DataFrame 相等
        tm.assert_frame_equal(result, expected)
    def test_isin_multiIndex(self):
        idx = MultiIndex.from_tuples(
            [
                (0, "a", "foo"),
                (0, "a", "bar"),
                (0, "b", "bar"),
                (0, "b", "baz"),
                (2, "a", "foo"),
                (2, "a", "bar"),
                (2, "c", "bar"),
                (2, "c", "baz"),
                (1, "b", "foo"),
                (1, "b", "bar"),
                (1, "c", "bar"),
                (1, "c", "baz"),
            ]
        )
        # 用元组列表创建一个多级索引对象 idx

        df1 = DataFrame({"A": np.ones(12), "B": np.zeros(12)}, index=idx)
        # 创建一个 DataFrame df1，包含两列 A 和 B，索引为 idx，数据初始化为 1 和 0

        df2 = DataFrame(
            {
                "A": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                "B": [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
            }
        )
        # 创建另一个 DataFrame df2，包含两列 A 和 B，数据为列表初始化

        # 对比普通索引
        expected = DataFrame(False, index=df1.index, columns=df1.columns)
        # 创建一个期望的 DataFrame，形状与 df1 相同，数据全为 False
        result = df1.isin(df2)
        # 使用 df2 对 df1 进行 isin 操作，返回一个布尔值 DataFrame
        tm.assert_frame_equal(result, expected)
        # 使用测试框架比较 result 和 expected 是否相等

        df2.index = idx
        # 修改 df2 的索引为 idx
        expected = df2.values.astype(bool)
        expected[:, 1] = ~expected[:, 1]
        expected = DataFrame(expected, columns=["A", "B"], index=idx)
        # 创建一个期望的 DataFrame，数据为 df2 的布尔值形式，第二列取反

        result = df1.isin(df2)
        # 再次使用 df2 对 df1 进行 isin 操作，返回一个布尔值 DataFrame
        tm.assert_frame_equal(result, expected)
        # 使用测试框架比较 result 和 expected 是否相等

    def test_isin_empty_datetimelike(self):
        # GH#15473
        df1_ts = DataFrame({"date": pd.to_datetime(["2014-01-01", "2014-01-02"])})
        # 创建一个包含日期时间数据的 DataFrame df1_ts

        df1_td = DataFrame({"date": [pd.Timedelta(1, "s"), pd.Timedelta(2, "s")]})
        # 创建一个包含时间增量数据的 DataFrame df1_td

        df2 = DataFrame({"date": []})
        df3 = DataFrame()
        # 创建两个空的 DataFrame df2 和 df3

        expected = DataFrame({"date": [False, False]})
        # 创建一个期望的 DataFrame，包含一个列 date，数据全为 False

        result = df1_ts.isin(df2)
        # 使用 df2 对 df1_ts 进行 isin 操作，返回一个布尔值 DataFrame
        tm.assert_frame_equal(result, expected)
        # 使用测试框架比较 result 和 expected 是否相等

        result = df1_ts.isin(df3)
        # 使用 df3 对 df1_ts 进行 isin 操作，返回一个布尔值 DataFrame
        tm.assert_frame_equal(result, expected)
        # 使用测试框架比较 result 和 expected 是否相等

        result = df1_td.isin(df2)
        # 使用 df2 对 df1_td 进行 isin 操作，返回一个布尔值 DataFrame
        tm.assert_frame_equal(result, expected)
        # 使用测试框架比较 result 和 expected 是否相等

        result = df1_td.isin(df3)
        # 使用 df3 对 df1_td 进行 isin 操作，返回一个布尔值 DataFrame
        tm.assert_frame_equal(result, expected)
        # 使用测试框架比较 result 和 expected 是否相等

    @pytest.mark.parametrize(
        "values",
        [
            DataFrame({"a": [1, 2, 3]}, dtype="category"),
            Series([1, 2, 3], dtype="category"),
        ],
    )
    def test_isin_category_frame(self, values):
        # GH#34256
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # 创建一个 DataFrame df，包含两列 a 和 b，数据为列表初始化

        expected = DataFrame({"a": [True, True, True], "b": [False, False, False]})
        # 创建一个期望的 DataFrame，包含两列 a 和 b，数据符合条件

        result = df.isin(values)
        # 使用 values 对 df 进行 isin 操作，返回一个布尔值 DataFrame
        tm.assert_frame_equal(result, expected)
        # 使用测试框架比较 result 和 expected 是否相等

    def test_isin_read_only(self):
        # https://github.com/pandas-dev/pandas/issues/37174
        arr = np.array([1, 2, 3])
        arr.setflags(write=False)
        # 创建一个 NumPy 数组 arr，并将其写入标志设置为 False

        df = DataFrame([1, 2, 3])
        # 创建一个包含数值列表的 DataFrame df

        result = df.isin(arr)
        # 使用 arr 对 df 进行 isin 操作，返回一个布尔值 DataFrame

        expected = DataFrame([True, True, True])
        # 创建一个期望的 DataFrame，数据全为 True

        tm.assert_frame_equal(result, expected)
        # 使用测试框架比较 result 和 expected 是否相等

    def test_isin_not_lossy(self):
        # GH 53514
        val = 1666880195890293744
        # 创建一个大整数 val

        df = DataFrame({"a": [val], "b": [1.0]})
        # 创建一个包含整数和浮点数的 DataFrame df

        result = df.isin([val])
        # 使用 val 对 df 进行 isin 操作，返回一个布尔值 DataFrame

        expected = DataFrame({"a": [True], "b": [False]})
        # 创建一个期望的 DataFrame，包含列 a 和 b，数据符合条件

        tm.assert_frame_equal(result, expected)
        # 使用测试框架比较 result 和 expected 是否相等
```