# `D:\src\scipysrc\pandas\pandas\tests\indexing\test_categorical.py`

```
import re  # 导入正则表达式模块

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 测试框架

import pandas.util._test_decorators as td  # 导入 pandas 内部测试装饰器模块

import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
from pandas import (  # 从 Pandas 中导入多个类和函数
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    Index,
    Interval,
    Series,
    Timedelta,
    Timestamp,
    option_context,
)
import pandas._testing as tm  # 导入 pandas 测试辅助模块


@pytest.fixture
def df():  # 定义一个名为 df 的 pytest fixture 函数
    return DataFrame(  # 返回一个 DataFrame 对象
        {
            "A": np.arange(6, dtype="int64"),  # 创建一个列名为 "A" 的整数列，值为 0 到 5
        },
        index=CategoricalIndex(  # 设置 DataFrame 的索引为 CategoricalIndex 类型
            list("aabbca"),  # 索引数据为列表 ['a', 'a', 'b', 'b', 'c', 'a']
            dtype=CategoricalDtype(list("cab")),  # 索引数据类型为 CategoricalDtype，类别为 ['c', 'a', 'b']
            name="B",  # 设置索引的名称为 "B"
        ),
    )


@pytest.fixture
def df2():  # 定义一个名为 df2 的 pytest fixture 函数
    return DataFrame(  # 返回一个 DataFrame 对象
        {
            "A": np.arange(6, dtype="int64"),  # 创建一个列名为 "A" 的整数列，值为 0 到 5
        },
        index=CategoricalIndex(  # 设置 DataFrame 的索引为 CategoricalIndex 类型
            list("aabbca"),  # 索引数据为列表 ['a', 'a', 'b', 'b', 'c', 'a']
            dtype=CategoricalDtype(list("cabe")),  # 索引数据类型为 CategoricalDtype，类别为 ['c', 'a', 'b', 'e']
            name="B",  # 设置索引的名称为 "B"
        ),
    )


class TestCategoricalIndex:  # 定义一个测试类 TestCategoricalIndex
    def test_loc_scalar(self, df):  # 定义测试方法 test_loc_scalar，接收 df fixture 作为参数
        dtype = CategoricalDtype(list("cab"))  # 创建一个 CategoricalDtype 对象，类别为 ['c', 'a', 'b']
        result = df.loc["a"]  # 根据索引 "a" 获取 DataFrame 的子集
        bidx = Series(list("aaa"), name="B").astype(dtype)  # 创建一个 Series 对象，索引为 ['a', 'a', 'a']，数据类型转换为 dtype
        assert bidx.dtype == dtype  # 断言 bidx 的数据类型与 dtype 相同

        expected = DataFrame({"A": [0, 1, 5]}, index=Index(bidx))  # 创建预期的 DataFrame 对象
        tm.assert_frame_equal(result, expected)  # 使用测试辅助模块 tm 进行 DataFrame 对象的相等断言

        df = df.copy()  # 复制 df DataFrame 对象
        df.loc["a"] = 20  # 将索引为 "a" 的行设置为值为 20
        bidx2 = Series(list("aabbca"), name="B").astype(dtype)  # 创建一个 Series 对象，索引为 ['a', 'a', 'b', 'b', 'c', 'a']，数据类型转换为 dtype
        assert bidx2.dtype == dtype  # 断言 bidx2 的数据类型与 dtype 相同
        expected = DataFrame(  # 创建预期的 DataFrame 对象
            {
                "A": [20, 20, 2, 3, 4, 20],  # 列 "A" 的数据
            },
            index=Index(bidx2),  # 设置索引为 bidx2
        )
        tm.assert_frame_equal(df, expected)  # 使用测试辅助模块 tm 进行 DataFrame 对象的相等断言

        # value not in the categories
        with pytest.raises(KeyError, match=r"^'d'$"):  # 使用 pytest 的异常断言，断言会抛出 KeyError 异常，异常信息匹配正则表达式 '^d$'
            df.loc["d"]  # 尝试获取索引为 "d" 的行数据

        df2 = df.copy()  # 复制 df DataFrame 对象
        expected = df2.copy()  # 复制 df2 DataFrame 对象
        expected.index = expected.index.astype(object)  # 将预期的 DataFrame 的索引类型转换为 object
        expected.loc["d"] = 10  # 将索引为 "d" 的行设置为值为 10
        df2.loc["d"] = 10  # 将索引为 "d" 的行设置为值为 10
        tm.assert_frame_equal(df2, expected)  # 使用测试辅助模块 tm 进行 DataFrame 对象的相等断言

    def test_loc_setitem_with_expansion_non_category(self, df):  # 定义测试方法 test_loc_setitem_with_expansion_non_category，接收 df fixture 作为参数
        # Setting-with-expansion with a new key "d" that is not among categories
        df.loc["a"] = 20  # 将索引为 "a" 的行设置为值为 20

        # Setting a new row on an existing column
        df3 = df.copy()  # 复制 df DataFrame 对象
        df3.loc["d", "A"] = 10  # 将索引为 "d"，列为 "A" 的位置设置为值为 10
        bidx3 = Index(list("aabbcad"), name="B")  # 创建一个 Index 对象，索引为 ['a', 'a', 'b', 'b', 'c', 'a', 'd']，名称为 "B"
        expected3 = DataFrame(  # 创建预期的 DataFrame 对象
            {
                "A": [20, 20, 2, 3, 4, 20, 10.0],  # 列 "A" 的数据
            },
            index=Index(bidx3),  # 设置索引为 bidx3
        )
        tm.assert_frame_equal(df3, expected3)  # 使用测试辅助模块 tm 进行 DataFrame 对象的相等断言

        # Setting a new row _and_ new column
        df4 = df.copy()  # 复制 df DataFrame 对象
        df4.loc["d", "C"] = 10  # 将索引为 "d"，列为 "C" 的位置设置为值为 10
        expected4 = DataFrame(  # 创建预期的 DataFrame 对象
            {
                "A": [20, 20, 2, 3, 4, 20, np.nan],  # 列 "A" 的数据
                "C": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10],  # 列 "C" 的数据
            },
            index=Index(bidx3),  # 设置索引为 bidx3
        )
        tm.assert_frame_equal(df4, expected4)  # 使用测试辅助模块 tm 进行 DataFrame 对象的相等断言

    def test_loc_getitem_scalar_non_category(self, df):  # 定义测试方法 test_loc_getitem_scalar_non_category，接收 df fixture 作为参数
        with pytest.raises(KeyError, match="^1$"):  # 使用 pytest 的异常断言，断言会抛出 KeyError 异常，异常信息匹配 "^1$"
            df.loc[1]  # 尝试获取索引为 1 的行数据
    def test_slicing(self):
        cat = Series(Categorical([1, 2, 3, 4]))
        # 对类别数据进行逆序切片
        reverse = cat[::-1]
        # 期望的 numpy 数组
        exp = np.array([4, 3, 2, 1], dtype=np.int64)
        # 断言逆序切片后的数组与期望数组相等
        tm.assert_numpy_array_equal(reverse.__array__(), exp)

        df = DataFrame({"value": (np.arange(100) + 1).astype("int64")})
        # 在 DataFrame 中创建分箱列 "D"
        df["D"] = pd.cut(df.value, bins=[0, 25, 50, 75, 100])

        # 期望的 Series，选择第 10 行的数据
        expected = Series([11, Interval(0, 25)], index=["value", "D"], name=10)
        result = df.iloc[10]
        # 断言结果 Series 与期望 Series 相等
        tm.assert_series_equal(result, expected)

        # 期望的 DataFrame，选择第 10 到 19 行的数据
        expected = DataFrame(
            {"value": np.arange(11, 21).astype("int64")},
            index=np.arange(10, 20).astype("int64"),
        )
        expected["D"] = pd.cut(expected.value, bins=[0, 25, 50, 75, 100])
        result = df.iloc[10:20]
        # 断言结果 DataFrame 与期望 DataFrame 相等
        tm.assert_frame_equal(result, expected)

        # 期望的 Series，选择索引为 8 的数据
        expected = Series([9, Interval(0, 25)], index=["value", "D"], name=8)
        result = df.loc[8]
        # 断言结果 Series 与期望 Series 相等
        tm.assert_series_equal(result, expected)

    def test_slicing_doc_examples(self):
        # GH 7918
        # 创建分类数据和索引
        cats = Categorical(
            ["a", "b", "b", "b", "c", "c", "c"], categories=["a", "b", "c"]
        )
        idx = Index(["h", "i", "j", "k", "l", "m", "n"])
        values = [1, 2, 2, 2, 3, 4, 5]
        df = DataFrame({"cats": cats, "values": values}, index=idx)

        # 选择第 2 到 3 行和所有列的数据
        result = df.iloc[2:4, :]
        expected = DataFrame(
            {
                "cats": Categorical(["b", "b"], categories=["a", "b", "c"]),
                "values": [2, 2],
            },
            index=["j", "k"],
        )
        # 断言结果 DataFrame 与期望 DataFrame 相等
        tm.assert_frame_equal(result, expected)

        # 选择第 2 到 3 行和所有列的数据类型
        result = df.iloc[2:4, :].dtypes
        expected = Series(["category", "int64"], ["cats", "values"], dtype=object)
        # 断言结果 Series 与期望 Series 相等
        tm.assert_series_equal(result, expected)

        # 选择索引为 "h" 到 "j" 的 "cats" 列数据
        result = df.loc["h":"j", "cats"]
        expected = Series(
            Categorical(["a", "b", "b"], categories=["a", "b", "c"]),
            index=["h", "i", "j"],
            name="cats",
        )
        # 断言结果 Series 与期望 Series 相等
        tm.assert_series_equal(result, expected)

        # 选择索引为 "h" 到 "j" 的 "cats" 列数据的 DataFrame
        result = df.loc["h":"j", df.columns[0:1]]
        expected = DataFrame(
            {"cats": Categorical(["a", "b", "b"], categories=["a", "b", "c"])},
            index=["h", "i", "j"],
        )
        # 断言结果 DataFrame 与期望 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_listlike_labels(self, df):
        # 使用标签列表选择数据
        result = df.loc[["c", "a"]]
        expected = df.iloc[[4, 0, 1, 5]]
        # 断言结果 DataFrame 与期望 DataFrame 相等，检查索引类型
        tm.assert_frame_equal(result, expected, check_index_type=True)

    def test_loc_getitem_listlike_unused_category(self, df2):
        # GH#37901：在索引的类别中但不在索引中的标签
        # 使用包含类别中但不在值中的元素的列表
        with pytest.raises(KeyError, match=re.escape("['e'] not in index")):
            df2.loc[["a", "b", "e"]]
    # 测试函数：测试在 DataFrame 中使用 loc 访问不存在于索引中但存在于分类中的元素
    def test_loc_getitem_label_unused_category(self, df2):
        # 断言期望抛出 KeyError 异常，异常信息应匹配正则表达式 "^'e'$"
        with pytest.raises(KeyError, match=r"^'e'$"):
            df2.loc["e"]

    # 测试函数：测试在 DataFrame 中使用 loc 访问包含但不完全存在于索引中的标签
    def test_loc_getitem_non_category(self, df2):
        # 断言期望抛出 KeyError 异常，异常信息应包含 "['d'] not in index" 的转义形式
        with pytest.raises(KeyError, match=re.escape("['d'] not in index")):
            df2.loc[["a", "d"]]

    # 测试函数：测试在 DataFrame 中使用 loc 设置一个存在于分类但不存在于索引中的标签
    def test_loc_setitem_expansion_label_unused_category(self, df2):
        # 复制 df2 DataFrame 到 df
        df = df2.copy()
        # 使用标签 "e" 对 df 进行设置
        df.loc["e"] = 20
        # 获取结果 DataFrame，索引包含 ["a", "b", "e"] 的分类索引
        result = df.loc[["a", "b", "e"]]
        # 期望结果 DataFrame，索引为 CategoricalIndex，包含 ["a", "b", "e"] 的分类索引，列 "A" 包含 [0, 1, 5, 2, 3, 20] 的数据
        exp_index = CategoricalIndex(list("aaabbe"), categories=list("cabe"), name="B")
        expected = DataFrame({"A": [0, 1, 5, 2, 3, 20]}, index=exp_index)
        # 使用 assert_frame_equal 进行结果比较
        tm.assert_frame_equal(result, expected)

    # 测试函数：测试在 DataFrame 中使用 loc 访问列表形式的数据类型
    def test_loc_listlike_dtypes(self):
        # GH 11586

        # 创建具有唯一分类和代码的索引
        index = CategoricalIndex(["a", "b", "c"])
        # 创建 DataFrame 包含两列 "A" 和 "B"，索引为上述分类索引
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=index)

        # 使用唯一切片
        res = df.loc[["a", "b"]]
        # 期望结果 DataFrame，索引为包含 ["a", "b"] 的分类索引，分类与 index 相同，列 "A" 包含 [1, 2] 的数据，列 "B" 包含 [4, 5] 的数据
        exp_index = CategoricalIndex(["a", "b"], categories=index.categories)
        exp = DataFrame({"A": [1, 2], "B": [4, 5]}, index=exp_index)
        # 使用 assert_frame_equal 进行结果比较，检查索引类型为 True
        tm.assert_frame_equal(res, exp, check_index_type=True)

        # 使用重复切片
        res = df.loc[["a", "a", "b"]]
        # 期望结果 DataFrame，索引为包含 ["a", "a", "b"] 的分类索引，分类与 index 相同，列 "A" 包含 [1, 1, 2] 的数据，列 "B" 包含 [4, 4, 5] 的数据
        exp_index = CategoricalIndex(["a", "a", "b"], categories=index.categories)
        exp = DataFrame({"A": [1, 1, 2], "B": [4, 4, 5]}, index=exp_index)
        # 使用 assert_frame_equal 进行结果比较，检查索引类型为 True
        tm.assert_frame_equal(res, exp, check_index_type=True)

        # 断言期望抛出 KeyError 异常，异常信息应包含 "['x'] not in index" 的转义形式
        with pytest.raises(KeyError, match=re.escape("['x'] not in index")):
            df.loc[["a", "x"]]

    # 测试函数：测试在 DataFrame 中使用 loc 访问具有重复分类和代码的数据类型
    def test_loc_listlike_dtypes_duplicated_categories_and_codes(self):
        # 创建具有重复分类和代码的索引
        index = CategoricalIndex(["a", "b", "a"])
        # 创建 DataFrame 包含两列 "A" 和 "B"，索引为上述分类索引
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=index)

        # 使用唯一切片
        res = df.loc[["a", "b"]]
        # 期望结果 DataFrame，索引为包含 ["a", "b"] 的分类索引，列 "A" 包含 [1, 3, 2] 的数据，列 "B" 包含 [4, 6, 5] 的数据
        exp = DataFrame(
            {"A": [1, 3, 2], "B": [4, 6, 5]}, index=CategoricalIndex(["a", "a", "b"])
        )
        # 使用 assert_frame_equal 进行结果比较，检查索引类型为 True
        tm.assert_frame_equal(res, exp, check_index_type=True)

        # 使用重复切片
        res = df.loc[["a", "a", "b"]]
        # 期望结果 DataFrame，索引为包含 ["a", "a", "b"] 的分类索引，列 "A" 包含 [1, 3, 1, 3, 2] 的数据，列 "B" 包含 [4, 6, 4, 6, 5] 的数据
        exp = DataFrame(
            {"A": [1, 3, 1, 3, 2], "B": [4, 6, 4, 6, 5]},
            index=CategoricalIndex(["a", "a", "a", "a", "b"]),
        )
        # 使用 assert_frame_equal 进行结果比较，检查索引类型为 True
        tm.assert_frame_equal(res, exp, check_index_type=True)

        # 断言期望抛出 KeyError 异常，异常信息应包含 "['x'] not in index" 的转义形式
        with pytest.raises(KeyError, match=re.escape("['x'] not in index")):
            df.loc[["a", "x"]]
    # 定义一个测试函数，用于测试处理未使用的类别的 loc 方法行为
    def test_loc_listlike_dtypes_unused_category(self):
        # 创建一个分类索引对象，包含字符串列表和指定的类别列表
        index = CategoricalIndex(["a", "b", "a", "c"], categories=list("abcde"))
        # 创建一个数据框，包含两列 A 和 B，以及上述的分类索引
        df = DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, index=index)

        # 使用 loc 方法选取索引中的特定值 'a' 和 'b'
        res = df.loc[["a", "b"]]
        # 创建预期的数据框，包含与 loc 选取相对应的值，并重新定义索引为包含所选值的分类索引
        exp = DataFrame(
            {"A": [1, 3, 2], "B": [5, 7, 6]},
            index=CategoricalIndex(["a", "a", "b"], categories=list("abcde")),
        )
        # 使用 pytest 的辅助函数验证 res 与 exp 是否相等，包括索引类型的检查
        tm.assert_frame_equal(res, exp, check_index_type=True)

        # 再次使用 loc 方法，但这次选择包含重复值的切片 ['a', 'a', 'b']
        res = df.loc[["a", "a", "b"]]
        # 创建预期的数据框，包含与 loc 选取相对应的值，同时重新定义索引为包含所选值的分类索引
        exp = DataFrame(
            {"A": [1, 3, 1, 3, 2], "B": [5, 7, 5, 7, 6]},
            index=CategoricalIndex(["a", "a", "a", "a", "b"], categories=list("abcde")),
        )
        # 使用 pytest 的辅助函数验证 res 与 exp 是否相等，包括索引类型的检查
        tm.assert_frame_equal(res, exp, check_index_type=True)

        # 使用 pytest 的断言验证在尝试使用不存在的键 'x' 时是否会引发 KeyError 异常
        with pytest.raises(KeyError, match=re.escape("['x'] not in index")):
            df.loc[["a", "x"]]

    # 定义一个测试函数，用于测试 loc 方法在处理未使用的类别时是否会引发 KeyError 异常
    def test_loc_getitem_listlike_unused_category_raises_keyerror(self):
        # 创建一个分类索引对象，包含字符串列表和指定的类别列表
        index = CategoricalIndex(["a", "b", "a", "c"], categories=list("abcde"))
        # 创建一个数据框，包含两列 A 和 B，以及上述的分类索引
        df = DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, index=index)

        # 使用 pytest 的断言验证在尝试使用不存在的键 'e' 时是否会引发 KeyError 异常
        with pytest.raises(KeyError, match="e"):
            df.loc["e"]

        # 使用 pytest 的断言验证在尝试使用包含不存在的键 'e' 的列表时是否会引发 KeyError 异常
        with pytest.raises(KeyError, match=re.escape("['e'] not in index")):
            df.loc[["a", "e"]]

    # 定义一个测试函数，用于测试 loc 方法在处理分类索引时的行为
    def test_ix_categorical_index(self):
        # 创建一个随机数据矩阵，指定索引和列名为列表 'ABC' 和 'XYZ'，并将其转换为数据框
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=list("ABC"),
            columns=list("XYZ"),
        )
        # 创建 df 的副本 cdf，并将其索引和列名分别转换为分类索引
        cdf = df.copy()
        cdf.index = CategoricalIndex(df.index)
        cdf.columns = CategoricalIndex(df.columns)

        # 使用 loc 方法选取分类索引中 'A' 行的数据，并与预期结果进行比较
        expect = Series(df.loc["A", :], index=cdf.columns, name="A")
        tm.assert_series_equal(cdf.loc["A", :], expect)

        # 使用 loc 方法选取列名为 'X' 的数据，并与预期结果进行比较
        expect = Series(df.loc[:, "X"], index=cdf.index, name="X")
        tm.assert_series_equal(cdf.loc[:, "X"], expect)

        # 使用 loc 方法选取分类索引中 'A' 和 'B' 行的数据，并与预期结果进行比较
        exp_index = CategoricalIndex(list("AB"), categories=["A", "B", "C"])
        expect = DataFrame(df.loc[["A", "B"], :], columns=cdf.columns, index=exp_index)
        tm.assert_frame_equal(cdf.loc[["A", "B"], :], expect)

        # 使用 loc 方法选取列名为 'X' 和 'Y' 的数据，并与预期结果进行比较
        exp_columns = CategoricalIndex(list("XY"), categories=["X", "Y", "Z"])
        expect = DataFrame(df.loc[:, ["X", "Y"]], index=cdf.index, columns=exp_columns)
        tm.assert_frame_equal(cdf.loc[:, ["X", "Y"]], expect)
    # 测试在使用非唯一索引的情况下的行为
    def test_ix_categorical_index_non_unique(self, infer_string):
        # 设置将来的推断字符串选项
        with option_context("future.infer_string", infer_string):
            # 创建一个 3x3 的随机数据帧，使用非唯一索引和列标签
            df = DataFrame(
                np.random.default_rng(2).standard_normal((3, 3)),
                index=list("ABA"),
                columns=list("XYX"),
            )
            # 创建数据帧的副本
            cdf = df.copy()
            # 将索引和列转换为分类索引
            cdf.index = CategoricalIndex(df.index)
            cdf.columns = CategoricalIndex(df.columns)

            # 预期的索引为分类索引 "AA"，其中类别为 ["A", "B"]
            exp_index = CategoricalIndex(list("AA"), categories=["A", "B"])
            # 创建预期结果，选择行"A"，列索引保持不变
            expect = DataFrame(df.loc["A", :], columns=cdf.columns, index=exp_index)
            tm.assert_frame_equal(cdf.loc["A", :], expect)

            # 预期的列为分类索引 "XX"，其中类别为 ["X", "Y"]
            exp_columns = CategoricalIndex(list("XX"), categories=["X", "Y"])
            # 创建预期结果，选择列 "X"，索引保持不变
            expect = DataFrame(df.loc[:, "X"], index=cdf.index, columns=exp_columns)
            tm.assert_frame_equal(cdf.loc[:, "X"], expect)

            # 创建预期结果，选择行 ["A", "B"]，列索引保持不变
            expect = DataFrame(
                df.loc[["A", "B"], :],
                columns=cdf.columns,
                index=CategoricalIndex(list("AAB")),
            )
            tm.assert_frame_equal(cdf.loc[["A", "B"], :], expect)

            # 创建预期结果，选择列 ["X", "Y"]，索引保持不变
            expect = DataFrame(
                df.loc[:, ["X", "Y"]],
                index=cdf.index,
                columns=CategoricalIndex(list("XXY")),
            )
            tm.assert_frame_equal(cdf.loc[:, ["X", "Y"]], expect)

    # 测试在分类索引上使用 loc 的切片
    def test_loc_slice(self, df):
        # GH9748
        msg = (
            "cannot do slice indexing on CategoricalIndex with these "
            r"indexers \[1\] of type int"
        )
        # 断言在切片索引时会引发类型错误，并匹配预期的消息
        with pytest.raises(TypeError, match=msg):
            df.loc[1:5]

        # 选择索引从 "b" 到 "c" 的行
        result = df.loc["b":"c"]
        expected = df.iloc[[2, 3, 4]]
        tm.assert_frame_equal(result, expected)

    # 测试在使用分类索引时的 loc 和 at 方法
    def test_loc_and_at_with_categorical_index(self):
        # GH 20629
        # 创建一个具有分类索引 ["A", "B", "C"] 的数据帧
        df = DataFrame(
            [[1, 2], [3, 4], [5, 6]], index=CategoricalIndex(["A", "B", "C"])
        )

        # 选择第一列的数据，并断言在索引 "A" 处的值为 1
        s = df[0]
        assert s.loc["A"] == 1
        assert s.at["A"] == 1

        # 断言在索引 "B" 处第二列的值为 4
        assert df.loc["B", 1] == 4
        assert df.at["B", 1] == 4
    @pytest.mark.parametrize(
        "idx_values",
        [
            # python types
            [1, 2, 3],               # List of integers [1, 2, 3]
            [-1, -2, -3],            # List of negative integers [-1, -2, -3]
            [1.5, 2.5, 3.5],         # List of floats [1.5, 2.5, 3.5]
            [-1.5, -2.5, -3.5],      # List of negative floats [-1.5, -2.5, -3.5]
            # numpy int/uint
            *(np.array([1, 2, 3], dtype=dtype) for dtype in tm.ALL_INT_NUMPY_DTYPES),  # Arrays of numpy integers with various dtypes
            # numpy floats
            *(np.array([1.5, 2.5, 3.5], dtype=dtyp) for dtyp in tm.FLOAT_NUMPY_DTYPES),  # Arrays of numpy floats with various dtypes
            # numpy object
            np.array([1, "b", 3.5], dtype=object),  # Array of mixed types: int, str, float
            # pandas scalars
            [Interval(1, 4), Interval(4, 6), Interval(6, 9)],  # List of pandas Interval objects
            [Timestamp(2019, 1, 1), Timestamp(2019, 2, 1), Timestamp(2019, 3, 1)],  # List of pandas Timestamp objects
            [Timedelta(1, "D"), Timedelta(2, "D"), Timedelta(3, "D")],  # List of pandas Timedelta objects
            # pandas Integer arrays
            *(pd.array([1, 2, 3], dtype=dtype) for dtype in tm.ALL_INT_EA_DTYPES),  # Arrays of pandas Integer dtype with various dtypes
            # other pandas arrays
            pd.IntervalIndex.from_breaks([1, 4, 6, 9]).array,  # Array from a pandas IntervalIndex
            pd.date_range("2019-01-01", periods=3).array,     # Array from a pandas date_range
            pd.timedelta_range(start="1D", periods=3).array,  # Array from a pandas timedelta_range
        ],
    )
    def test_loc_getitem_with_non_string_categories(self, idx_values, ordered):
        # GH-17569: Test case for loc selection on DataFrame with CategoricalIndex
    
        # Create a CategoricalIndex with given index values and order
        cat_idx = CategoricalIndex(idx_values, ordered=ordered)
    
        # Create a DataFrame with a single column "A" indexed by cat_idx
        df = DataFrame({"A": ["foo", "bar", "baz"]}, index=cat_idx)
    
        # Define a slice using the first two index values
        sl = slice(idx_values[0], idx_values[1])
    
        # Test scalar selection using loc
        result = df.loc[idx_values[0]]
        expected = Series(["foo"], index=["A"], name=idx_values[0])
        tm.assert_series_equal(result, expected)
    
        # Test list selection using loc
        result = df.loc[idx_values[:2]]
        expected = DataFrame(["foo", "bar"], index=cat_idx[:2], columns=["A"])
        tm.assert_frame_equal(result, expected)
    
        # Test slice selection using loc
        result = df.loc[sl]
        expected = DataFrame(["foo", "bar"], index=cat_idx[:2], columns=["A"])
        tm.assert_frame_equal(result, expected)
    
        # Test scalar assignment using loc
        result = df.copy()
        result.loc[idx_values[0]] = "qux"
        expected = DataFrame({"A": ["qux", "bar", "baz"]}, index=cat_idx)
        tm.assert_frame_equal(result, expected)
    
        # Test list assignment using loc
        result = df.copy()
        result.loc[idx_values[:2], "A"] = ["qux", "qux2"]
        expected = DataFrame({"A": ["qux", "qux2", "baz"]}, index=cat_idx)
        tm.assert_frame_equal(result, expected)
    
        # Test slice assignment using loc
        result = df.copy()
        result.loc[sl, "A"] = ["qux", "qux2"]
        expected = DataFrame({"A": ["qux", "qux2", "baz"]}, index=cat_idx)
        tm.assert_frame_equal(result, expected)
    
    
    def test_getitem_categorical_with_nan():
        # GH#41933: Test case for accessing DataFrame and Series with NaN in CategoricalIndex
    
        # Create a CategoricalIndex with categories 'A', 'B', and NaN
        ci = CategoricalIndex(["A", "B", np.nan])
    
        # Create a Series indexed by ci
        ser = Series(range(3), index=ci)
    
        # Assert accessing series with NaN returns correct value
        assert ser[np.nan] == 2
        assert ser.loc[np.nan] == 2
    
        # Create a DataFrame with ser as the only column
        df = DataFrame(ser)
    
        # Assert accessing DataFrame with NaN using loc returns correct value
        assert df.loc[np.nan, 0] == 2
        assert df.loc[np.nan][0] == 2
```