# `D:\src\scipysrc\pandas\pandas\tests\frame\test_nonunique_indexes.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试

import pandas as pd  # 导入 Pandas 库，用于数据操作
from pandas import (  # 从 Pandas 中导入 DataFrame、Series、date_range 等对象
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm  # 导入 Pandas 测试模块


class TestDataFrameNonuniqueIndexes:
    def test_setattr_columns_vs_construct_with_columns(self):
        # assignment
        # GH 3687
        arr = np.random.default_rng(2).standard_normal((3, 2))  # 生成一个 3x2 的标准正态分布随机数组
        idx = list(range(2))  # 创建一个包含两个整数的列表
        df = DataFrame(arr, columns=["A", "A"])  # 创建一个 DataFrame，列名为 ["A", "A"]
        df.columns = idx  # 修改 DataFrame 的列名为 idx
        expected = DataFrame(arr, columns=idx)  # 创建一个期望的 DataFrame，列名为 idx
        tm.assert_frame_equal(df, expected)  # 使用测试工具检查 df 和 expected 是否相等

    def test_setattr_columns_vs_construct_with_columns_datetimeindx(self):
        idx = date_range("20130101", periods=4, freq="QE-NOV")  # 创建一个 DatetimeIndex
        df = DataFrame(  # 创建一个 DataFrame
            [[1, 1, 1, 5], [1, 1, 2, 5], [2, 1, 3, 5]], columns=["a", "a", "a", "a"]
        )
        df.columns = idx  # 修改 DataFrame 的列名为 idx
        expected = DataFrame([[1, 1, 1, 5], [1, 1, 2, 5], [2, 1, 3, 5]], columns=idx)  # 创建一个期望的 DataFrame，列名为 idx
        tm.assert_frame_equal(df, expected)  # 使用测试工具检查 df 和 expected 是否相等

    def test_dup_across_dtypes(self):
        # dup across dtypes
        df = DataFrame(  # 创建一个 DataFrame
            [[1, 1, 1.0, 5], [1, 1, 2.0, 5], [2, 1, 3.0, 5]],
            columns=["foo", "bar", "foo", "hello"],
        )

        df["foo2"] = 7.0  # 在 DataFrame 中添加名为 "foo2" 的列，并赋值为 7.0
        expected = DataFrame(  # 创建一个期望的 DataFrame
            [[1, 1, 1.0, 5, 7.0], [1, 1, 2.0, 5, 7.0], [2, 1, 3.0, 5, 7.0]],
            columns=["foo", "bar", "foo", "hello", "foo2"],
        )
        tm.assert_frame_equal(df, expected)  # 使用测试工具检查 df 和 expected 是否相等

        result = df["foo"]  # 从 DataFrame 中选取 "foo" 列
        expected = DataFrame([[1, 1.0], [1, 2.0], [2, 3.0]], columns=["foo", "foo"])  # 创建一个期望的 DataFrame，列名为 ["foo", "foo"]
        tm.assert_frame_equal(result, expected)  # 使用测试工具检查 result 和 expected 是否相等

        # multiple replacements
        df["foo"] = "string"  # 将 DataFrame 中的 "foo" 列的所有值替换为字符串 "string"
        expected = DataFrame(  # 创建一个期望的 DataFrame
            [
                ["string", 1, "string", 5, 7.0],
                ["string", 1, "string", 5, 7.0],
                ["string", 1, "string", 5, 7.0],
            ],
            columns=["foo", "bar", "foo", "hello", "foo2"],
        )
        tm.assert_frame_equal(df, expected)  # 使用测试工具检查 df 和 expected 是否相等

        del df["foo"]  # 从 DataFrame 中删除 "foo" 列
        expected = DataFrame(  # 创建一个期望的 DataFrame
            [[1, 5, 7.0], [1, 5, 7.0], [1, 5, 7.0]], columns=["bar", "hello", "foo2"]
        )
        tm.assert_frame_equal(df, expected)  # 使用测试工具检查 df 和 expected 是否相等

    def test_column_dups_indexes(self):
        # check column dups with index equal and not equal to df's index
        df = DataFrame(  # 创建一个 DataFrame
            np.random.default_rng(2).standard_normal((5, 3)),  # 生成一个 5x3 的标准正态分布随机数组
            index=["a", "b", "c", "d", "e"],  # 设置 DataFrame 的行索引
            columns=["A", "B", "A"],  # 设置 DataFrame 的列名，包含重复的 "A"
        )
        for index in [df.index, pd.Index(list("edcba"))]:  # 遍历不同的索引
            this_df = df.copy()  # 复制当前的 DataFrame
            expected_ser = Series(index.values, index=this_df.index)  # 创建一个期望的 Series，使用 index 和当前 DataFrame 的索引
            expected_df = DataFrame(  # 创建一个期望的 DataFrame
                {"A": expected_ser, "B": this_df["B"]},
                columns=["A", "B", "A"],
            )
            this_df["A"] = index  # 将 DataFrame 的 "A" 列设置为当前的索引
            tm.assert_frame_equal(this_df, expected_df)  # 使用测试工具检查 this_df 和 expected_df 是否相等
    def test_changing_dtypes_with_duplicate_columns(self):
        # 定义一个测试函数，用于测试在存在重复列名时改变数据类型的情况
        # 使用 NumPy 随机数生成器创建一个 5x2 的数据框，列名为 ["that", "that"]
        # GH 6120
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=["that", "that"]
        )
        # 创建预期结果，期望所有值为 1.0，索引为 [0, 1, 2, 3, 4]，列名为 ["that", "that"]
        expected = DataFrame(1.0, index=range(5), columns=["that", "that"])

        # 将 "that" 列的所有值设置为 1.0
        df["that"] = 1.0
        # 断言数据框 df 与预期结果 expected 相等
        tm.assert_frame_equal(df, expected)

        # 使用 NumPy 随机数生成器创建一个 5x2 的数据框，列名为 ["that", "that"]
        df = DataFrame(
            np.random.default_rng(2).random((5, 2)), columns=["that", "that"]
        )
        # 创建预期结果，期望所有值为 1，索引为 [0, 1, 2, 3, 4]，列名为 ["that", "that"]
        expected = DataFrame(1, index=range(5), columns=["that", "that"])

        # 将 "that" 列的所有值设置为 1
        df["that"] = 1
        # 断言数据框 df 与预期结果 expected 相等
        tm.assert_frame_equal(df, expected)

    def test_dup_columns_comparisons(self):
        # 定义一个测试函数，用于测试重复列名的比较操作
        df1 = DataFrame([[1, 2], [2, np.nan], [3, 4], [4, 4]], columns=["A", "B"])
        df2 = DataFrame([[0, 1], [2, 4], [2, np.nan], [4, 5]], columns=["A", "A"])

        # 尝试比较两个不同列标签的数据框，预期引发 ValueError，匹配特定的错误信息
        msg = (
            r"Can only compare identically-labeled \(both index and columns\) "
            "DataFrame objects"
        )
        with pytest.raises(ValueError, match=msg):
            df1 == df2

        # 使用 df1 的索引结构重新索引 df2，然后比较这两个数据框
        df1r = df1.reindex_like(df2)
        result = df1r == df2
        # 创建预期结果，期望比较后的数据框结构与值为 [[False, True], [True, False], [False, False], [True, False]]，列名为 ["A", "A"]
        expected = DataFrame(
            [[False, True], [True, False], [False, False], [True, False]],
            columns=["A", "A"],
        )
        # 断言比较后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    def test_mixed_column_selection(self):
        # 定义一个测试函数，用于测试混合列选择
        # GH 5639
        dfbool = DataFrame(
            {
                "one": Series([True, True, False], index=["a", "b", "c"]),
                "two": Series([False, False, True, False], index=["a", "b", "c", "d"]),
                "three": Series([False, True, True, True], index=["a", "b", "c", "d"]),
            }
        )
        # 创建预期结果，合并 dfbool 中的 "one"、"three"、"one" 列，列名为 ["one", "three", "one"]
        expected = pd.concat([dfbool["one"], dfbool["three"], dfbool["one"]], axis=1)
        # 选择 dfbool 中的 ["one", "three", "one"] 列
        result = dfbool[["one", "three", "one"]]
        # 断言选择的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    def test_multi_axis_dups(self):
        # 定义一个测试函数，用于测试多轴重复
        # GH 6121
        df = DataFrame(
            np.arange(25.0).reshape(5, 5),
            index=["a", "b", "c", "d", "e"],
            columns=["A", "B", "C", "D", "E"],
        )
        # 复制 df 中的 ["A", "C", "A"] 列，并命名为 z
        z = df[["A", "C", "A"]].copy()
        # 创建预期结果，选择 z 中的 ["a", "c", "a"] 行
        expected = z.loc[["a", "c", "a"]]

        df = DataFrame(
            np.arange(25.0).reshape(5, 5),
            index=["a", "b", "c", "d", "e"],
            columns=["A", "B", "C", "D", "E"],
        )
        # 选择 df 中的 ["A", "C", "A"] 列
        z = df[["A", "C", "A"]]
        # 选择 z 中的 ["a", "c", "a"] 行
        result = z.loc[["a", "c", "a"]]
        # 断言选择的结果与预期结果相等
        tm.assert_frame_equal(result, expected)
    def test_columns_with_dups(self):
        # GH 3468 related

        # 创建一个包含重复列名的DataFrame，进行列名修改测试
        df = DataFrame([[1, 2]], columns=["a", "a"])
        # 修改列名，验证列名修改后的DataFrame是否符合预期
        df.columns = ["a", "a.1"]
        expected = DataFrame([[1, 2]], columns=["a", "a.1"])
        # 使用assert_frame_equal方法比较DataFrame，确认修改后的列名是否正确
        tm.assert_frame_equal(df, expected)

        # 创建另一个包含重复列名的DataFrame，进行列名修改测试
        df = DataFrame([[1, 2, 3]], columns=["b", "a", "a"])
        # 修改列名，验证列名修改后的DataFrame是否符合预期
        df.columns = ["b", "a", "a.1"]
        expected = DataFrame([[1, 2, 3]], columns=["b", "a", "a.1"])
        # 使用assert_frame_equal方法比较DataFrame，确认修改后的列名是否正确
        tm.assert_frame_equal(df, expected)

    def test_columns_with_dup_index(self):
        # 测试包含重复索引的DataFrame，进行列名修改测试
        df = DataFrame([[1, 2]], columns=["a", "a"])
        # 修改列名，验证列名修改后的DataFrame是否符合预期
        df.columns = ["b", "b"]
        expected = DataFrame([[1, 2]], columns=["b", "b"])
        # 使用assert_frame_equal方法比较DataFrame，确认修改后的列名是否正确
        tm.assert_frame_equal(df, expected)

    def test_multi_dtype(self):
        # 测试包含多种数据类型的DataFrame，进行列名修改测试
        df = DataFrame(
            [[1, 2, 1.0, 2.0, 3.0, "foo", "bar"]],
            columns=["a", "a", "b", "b", "d", "c", "c"],
        )
        # 修改列名，验证列名修改后的DataFrame是否符合预期
        df.columns = list("ABCDEFG")
        expected = DataFrame(
            [[1, 2, 1.0, 2.0, 3.0, "foo", "bar"]], columns=list("ABCDEFG")
        )
        # 使用assert_frame_equal方法比较DataFrame，确认修改后的列名是否正确
        tm.assert_frame_equal(df, expected)

    def test_multi_dtype2(self):
        # 测试包含多种数据类型且有重复列名的DataFrame，进行列名修改测试
        df = DataFrame([[1, 2, "foo", "bar"]], columns=["a", "a", "a", "a"])
        # 修改列名，验证列名修改后的DataFrame是否符合预期
        df.columns = ["a", "a.1", "a.2", "a.3"]
        expected = DataFrame([[1, 2, "foo", "bar"]], columns=["a", "a.1", "a.2", "a.3"])
        # 使用assert_frame_equal方法比较DataFrame，确认修改后的列名是否正确
        tm.assert_frame_equal(df, expected)

    def test_dups_across_blocks(self):
        # 测试包含不同数据类型的DataFrame合并后，进行列名访问测试
        df_float = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), dtype="float64"
        )
        df_int = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)).astype("int64")
        )
        df_bool = DataFrame(True, index=df_float.index, columns=df_float.columns)
        df_object = DataFrame("foo", index=df_float.index, columns=df_float.columns)
        df_dt = DataFrame(
            pd.Timestamp("20010101"), index=df_float.index, columns=df_float.columns
        )
        # 将不同类型的DataFrame按列连接成一个新的DataFrame
        df = pd.concat([df_float, df_int, df_bool, df_object, df_dt], axis=1)

        # 断言DataFrame内部数据块数量与列数相等
        assert len(df._mgr.blknos) == len(df.columns)
        assert len(df._mgr.blklocs) == len(df.columns)

        # 使用iloc方法访问DataFrame的每一列数据
        # 遍历DataFrame的每一列，不执行其他操作
        for i in range(len(df.columns)):
            df.iloc[:, i]

    def test_dup_columns_across_dtype(self):
        # 测试跨数据类型存在重复列名的DataFrame，进行列名修改测试
        vals = [[1, -1, 2.0], [2, -2, 3.0]]
        rs = DataFrame(vals, columns=["A", "A", "B"])
        xp = DataFrame(vals)
        xp.columns = ["A", "A", "B"]
        # 使用assert_frame_equal方法比较两个DataFrame，确认列名修改后是否符合预期
        tm.assert_frame_equal(rs, xp)
    # 定义测试方法，用于测试按索引设置值的功能
    def test_set_value_by_index(self):
        # 设置警告信息变量为 None
        warn = None
        # 定义警告消息字符串
        msg = "will attempt to set the values inplace"

        # 创建一个 3x3 的数据框，其中元素为 0 到 8 的整数，转置后赋值给数据框 df
        df = DataFrame(np.arange(9).reshape(3, 3).T)
        # 将数据框的列名设置为 "AAA"
        df.columns = list("AAA")
        # 复制 df 的第三列，作为预期结果
        expected = df.iloc[:, 2].copy()

        # 使用上下文管理器断言设置值操作会产生警告，并匹配特定的警告消息
        with tm.assert_produces_warning(warn, match=msg):
            # 设置数据框 df 的第一列的所有元素为 3
            df.iloc[:, 0] = 3
        # 断言数据框 df 的第三列与预期结果相等
        tm.assert_series_equal(df.iloc[:, 2], expected)

        # 创建一个 3x3 的数据框，其中元素为 0 到 8 的整数，转置后赋值给数据框 df
        df = DataFrame(np.arange(9).reshape(3, 3).T)
        # 将数据框的列名设置为 [2, 2.0, '2']
        df.columns = [2, float(2), str(2)]
        # 复制 df 的第二列，作为预期结果
        expected = df.iloc[:, 1].copy()

        # 使用上下文管理器断言设置值操作会产生警告，并匹配特定的警告消息
        with tm.assert_produces_warning(warn, match=msg):
            # 设置数据框 df 的第一列的所有元素为 3
            df.iloc[:, 0] = 3
        # 断言数据框 df 的第二列与预期结果相等
        tm.assert_series_equal(df.iloc[:, 1], expected)
```