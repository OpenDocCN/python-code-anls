# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_update.py`

```
    import numpy as np
    # 导入 NumPy 库，用于处理数值数据
    import pytest
    # 导入 Pytest 库，用于编写和运行测试用例

    import pandas as pd
    # 导入 Pandas 库，用于数据操作和分析
    from pandas import (
        DataFrame,
        Series,
        date_range,
    )
    # 从 Pandas 中导入 DataFrame、Series 和 date_range 函数

    import pandas._testing as tm
    # 导入 Pandas 内部测试工具模块，用于测试框架中的断言等功能


    class TestDataFrameUpdate:
        def test_update_nan(self):
            # #15593 #15617
            # 创建 DataFrame df1 包含数字和日期范围
            df1 = DataFrame({"A": [1.0, 2, 3], "B": date_range("2000", periods=3)})
            # 创建 DataFrame df2 包含空值和数字
            df2 = DataFrame({"A": [None, 2, 3]})
            # 复制 df1 以备后用
            expected = df1.copy()
            # 使用 df2 更新 df1，保留未覆盖的值
            df1.update(df2, overwrite=False)

            tm.assert_frame_equal(df1, expected)
            # 断言 df1 和 expected 的内容相等

            # 创建第二组数据进行更新测试
            df1 = DataFrame({"A": [1.0, None, 3], "B": date_range("2000", periods=3)})
            df2 = DataFrame({"A": [None, 2, 3]})
            expected = DataFrame({"A": [1.0, 2, 3], "B": date_range("2000", periods=3)})
            df1.update(df2, overwrite=False)

            tm.assert_frame_equal(df1, expected)

        def test_update(self):
            # 创建包含 NaN 值的 DataFrame
            df = DataFrame(
                [[1.5, np.nan, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
            )

            other = DataFrame([[3.6, 2.0, np.nan], [np.nan, np.nan, 7]], index=[1, 3])
            # 使用 other DataFrame 更新 df
            df.update(other)

            expected = DataFrame(
                [[1.5, np.nan, 3], [3.6, 2, 3], [1.5, np.nan, 3], [1.5, np.nan, 7.0]]
            )
            tm.assert_frame_equal(df, expected)

        def test_update_dtypes(self):
            # gh 3016
            # 创建包含不同数据类型的 DataFrame
            df = DataFrame(
                [[1.0, 2.0, False, True], [4.0, 5.0, True, False]],
                columns=["A", "B", "bool1", "bool2"],
            )

            other = DataFrame([[45, 45]], index=[0], columns=["A", "B"])
            # 使用 other DataFrame 更新 df
            df.update(other)

            expected = DataFrame(
                [[45.0, 45.0, False, True], [4.0, 5.0, True, False]],
                columns=["A", "B", "bool1", "bool2"],
            )
            tm.assert_frame_equal(df, expected)

        def test_update_nooverwrite(self):
            # 创建包含 NaN 值的 DataFrame
            df = DataFrame(
                [[1.5, np.nan, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
            )

            other = DataFrame([[3.6, 2.0, np.nan], [np.nan, np.nan, 7]], index=[1, 3])
            # 使用 other DataFrame 更新 df，但不覆盖已存在的值
            df.update(other, overwrite=False)

            expected = DataFrame(
                [[1.5, np.nan, 3], [1.5, 2, 3], [1.5, np.nan, 3], [1.5, np.nan, 3.0]]
            )
            tm.assert_frame_equal(df, expected)

        def test_update_filtered(self):
            # 创建包含 NaN 值的 DataFrame
            df = DataFrame(
                [[1.5, np.nan, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
            )

            other = DataFrame([[3.6, 2.0, np.nan], [np.nan, np.nan, 7]], index=[1, 3])
            # 使用 other DataFrame 更新 df，但仅对大于 2 的值进行更新
            df.update(other, filter_func=lambda x: x > 2)

            expected = DataFrame(
                [[1.5, np.nan, 3], [1.5, np.nan, 3], [1.5, np.nan, 3], [1.5, np.nan, 7.0]]
            )
            tm.assert_frame_equal(df, expected)

        @pytest.mark.parametrize(
            "bad_kwarg, exception, msg",
            [
                # 错误参数应为 'ignore' 或 'raise'
                ({"errors": "something"}, ValueError, "The parameter errors must.*"),
                ({"join": "inner"}, NotImplementedError, "Only left join is supported"),
            ],
        )
    # 定义一个测试函数，用于测试在给定错误参数时是否会引发异常
    def test_update_raise_bad_parameter(self, bad_kwarg, exception, msg):
        # 创建一个包含单个数据框的数据帧，数据包括一个包含浮点数和整数的列表
        df = DataFrame([[1.5, 1, 3.0]])
        # 使用 pytest 的 assert 语句来检查是否引发了特定类型的异常，并匹配特定的消息
        with pytest.raises(exception, match=msg):
            # 调用数据帧的 update 方法，并传入错误的关键字参数
            df.update(df, **bad_kwarg)

    # 定义一个测试函数，测试在数据重叠时是否引发值错误异常
    def test_update_raise_on_overlap(self):
        # 创建一个包含多行数据的数据帧，包括 NaN 值和整数的列表
        df = DataFrame(
            [[1.5, 1, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
        )

        # 创建另一个数据帧，包含 NaN 值的索引和列，用于更新原始数据帧
        other = DataFrame([[2.0, np.nan], [np.nan, 7]], index=[1, 3], columns=[1, 2])
        # 使用 pytest 的 assert 语句来检查是否引发了值错误异常，并匹配特定的消息
        with pytest.raises(ValueError, match="Data overlaps"):
            # 调用数据帧的 update 方法，并设置错误处理方式为 "raise"
            df.update(other, errors="raise")

    # 定义一个测试函数，测试从非数据帧对象更新时的行为
    def test_update_from_non_df(self):
        # 创建一个字典，包含多个系列对象，每个系列对象包含整数数据
        d = {"a": Series([1, 2, 3, 4]), "b": Series([5, 6, 7, 8])}
        # 用字典创建一个数据帧
        df = DataFrame(d)

        # 修改字典中的 "a" 系列的数据
        d["a"] = Series([5, 6, 7, 8])
        # 使用数据帧的 update 方法，从修改后的字典更新数据帧
        df.update(d)

        # 创建预期的数据帧，从修改后的字典创建
        expected = DataFrame(d)

        # 使用测试工具类中的 assert_frame_equal 方法来比较两个数据帧是否相等
        tm.assert_frame_equal(df, expected)

        # 创建一个字典，包含多个整数列表，而不是系列对象
        d = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
        # 用字典创建一个数据帧
        df = DataFrame(d)

        # 修改字典中的 "a" 列的数据
        d["a"] = [5, 6, 7, 8]
        # 使用数据帧的 update 方法，从修改后的字典更新数据帧
        df.update(d)

        # 创建预期的数据帧，从修改后的字典创建
        expected = DataFrame(d)

        # 使用测试工具类中的 assert_frame_equal 方法来比较两个数据帧是否相等
        tm.assert_frame_equal(df, expected)

    # 定义一个测试函数，测试在包含带时区的日期时间时的行为
    def test_update_datetime_tz(self):
        # 创建包含带时区的日期时间的数据帧
        result = DataFrame([pd.Timestamp("2019", tz="UTC")])
        # 使用测试工具类中的 assert_produces_warning 方法，来确保在更新时不产生警告
        with tm.assert_produces_warning(None):
            # 调用数据帧的 update 方法，并传入自身来更新
            result.update(result)
        # 创建预期的数据帧，与原始数据帧相同
        expected = DataFrame([pd.Timestamp("2019", tz="UTC")])
        # 使用测试工具类中的 assert_frame_equal 方法来比较两个数据帧是否相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，测试在包含带时区的日期时间时的原地修改行为
    def test_update_datetime_tz_in_place(self):
        # 创建包含带时区的日期时间的数据帧
        result = DataFrame([pd.Timestamp("2019", tz="UTC")])
        # 复制原始数据帧
        orig = result.copy()
        # 创建数据帧的切片视图
        view = result[:]
        # 使用数据帧的 update 方法，传入增加一天后的数据来更新
        result.update(result + pd.Timedelta(days=1))
        # 创建预期的数据帧，包含增加一天后的日期时间
        expected = DataFrame([pd.Timestamp("2019-01-02", tz="UTC")])
        # 使用测试工具类中的 assert_frame_equal 方法来比较两个数据帧是否相等
        tm.assert_frame_equal(result, expected)
        # 使用测试工具类中的 assert_frame_equal 方法来比较视图和原始数据帧是否相等
        tm.assert_frame_equal(view, orig)

    # 定义一个测试函数，测试在包含不同数据类型时的行为
    def test_update_with_different_dtype(self):
        # 创建包含整数和 NaN 值的数据帧
        df = DataFrame({"a": [1, 3], "b": [np.nan, 2]})
        # 添加一个 NaN 值的新列 'c'
        df["c"] = np.nan
        # 使用测试工具类中的 assert_produces_warning 方法，确保在更新时产生 FutureWarning 警告
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            # 调用数据帧的 update 方法，从包含字符串的系列更新新列 'c'
            df.update({"c": Series(["foo"], index=[0])})

        # 创建预期的数据帧，包含字符串值的新列 'c'
        expected = DataFrame(
            {
                "a": [1, 3],
                "b": [np.nan, 2],
                "c": Series(["foo", np.nan], dtype="object"),
            }
        )
        # 使用测试工具类中的 assert_frame_equal 方法来比较两个数据帧是否相等
        tm.assert_frame_equal(df, expected)

    # 定义一个测试函数，测试在修改视图时是否正确更新数据帧
    def test_update_modify_view(self, using_infer_string):
        # 创建包含字符串和 NaN 值的数据帧 'df'
        df = DataFrame({"A": ["1", np.nan], "B": ["100", np.nan]})
        # 创建另一个数据帧 'df2'，包含不同的字符串和 NaN 值
        df2 = DataFrame({"A": ["a", "x"], "B": ["100", "200"]})
        # 复制 'df2' 以备后续比较使用
        df2_orig = df2.copy()
        # 创建 'df2' 的切片视图
        result_view = df2[:]
        # 使用数据帧 'df2' 的 update 方法，从数据帧 'df' 更新数据
        df2.update(df)

        # 创建预期的数据帧，'A' 列将保留原始值，'B' 列将更新为 "200"
        expected = DataFrame({"A": ["1", "x"], "B": ["100", "200"]})
        # 使用测试工具类中的 assert_frame_equal 方法来比较两个数据帧是否相等
        tm.assert_frame_equal(df2, expected)
        # 使用测试工具类中的 assert_frame_equal 方法来比较视图和原始数据帧是否相等
        tm.assert_frame_equal(result_view, df2_orig)
    # 定义一个测试方法，用于测试在DataFrame中使用update方法更新日期时间列并创建新列
    def test_update_dt_column_with_NaT_create_column(self):
        # GH#16713: GitHub issue标识号，用于跟踪相关问题
        df = DataFrame({"A": [1, None], "B": [pd.NaT, pd.to_datetime("2016-01-01")]})
        # 创建另一个DataFrame对象df2，用于更新原始DataFrame df
        df2 = DataFrame({"A": [2, 3]})
        # 使用update方法，将df2中的数据合并到df中，但不覆盖已有值
        df.update(df2, overwrite=False)
        # 期望的结果DataFrame，用于与更新后的df进行比较
        expected = DataFrame(
            {"A": [1.0, 3.0], "B": [pd.NaT, pd.to_datetime("2016-01-01")]}
        )
        # 使用测试工具tm.assert_frame_equal比较df和期望的expected是否相同
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "value_df, value_other, dtype",
        [
            (True, False, bool),
            (1, 2, int),
            (1.0, 2.0, float),
            (1.0 + 1j, 2.0 + 2j, complex),
            (np.uint64(1), np.uint(2), np.dtype("ubyte")),
            (np.uint64(1), np.uint(2), np.dtype("intc")),
            ("a", "b", pd.StringDtype()),
            (
                pd.to_timedelta("1 ms"),
                pd.to_timedelta("2 ms"),
                np.dtype("timedelta64[ns]"),
            ),
            (
                np.datetime64("2000-01-01T00:00:00"),
                np.datetime64("2000-01-02T00:00:00"),
                np.dtype("datetime64[ns]"),
            ),
        ],
    )
    # 使用pytest的参数化装饰器，定义一个测试方法，测试update方法在保留数据类型的情况下更新DataFrame
    def test_update_preserve_dtype(self, value_df, value_other, dtype):
        # GH#55509: GitHub issue标识号，用于跟踪相关问题
        # 创建一个DataFrame对象df，包含一个列"a"，值为value_df的复制，索引为[1, 2]，数据类型为dtype
        df = DataFrame({"a": [value_df] * 2}, index=[1, 2], dtype=dtype)
        # 创建另一个DataFrame对象other，包含一个列"a"，值为value_other，索引为[1]，数据类型为dtype
        other = DataFrame({"a": [value_other]}, index=[1], dtype=dtype)
        # 期望的结果DataFrame，用于与更新后的df进行比较
        expected = DataFrame({"a": [value_other, value_df]}, index=[1, 2], dtype=dtype)
        # 使用update方法，将other中的数据合并到df中
        df.update(other)
        # 使用测试工具tm.assert_frame_equal比较df和期望的expected是否相同
        tm.assert_frame_equal(df, expected)

    # 定义一个测试方法，测试在DataFrame中使用update方法时，如果出现重复的索引会抛出ValueError异常
    def test_update_raises_on_duplicate_argument_index(self):
        # GH#55509: GitHub issue标识号，用于跟踪相关问题
        df = DataFrame({"a": [1, 1]}, index=[1, 2])
        other = DataFrame({"a": [2, 3]}, index=[1, 1])
        # 使用pytest的raises方法，期望抛出ValueError异常，异常信息中包含"duplicate index"
        with pytest.raises(ValueError, match="duplicate index"):
            df.update(other)

    # 定义一个测试方法，测试在DataFrame中使用update方法时，如果两个DataFrame没有交集会抛出ValueError异常
    def test_update_raises_without_intersection(self):
        # GH#55509: GitHub issue标识号，用于跟踪相关问题
        df = DataFrame({"a": [1]}, index=[1])
        other = DataFrame({"a": [2]}, index=[2])
        # 使用pytest的raises方法，期望抛出ValueError异常，异常信息中包含"no intersection"
        with pytest.raises(ValueError, match="no intersection"):
            df.update(other)

    # 定义一个测试方法，测试在DataFrame中使用update方法时，处理重复索引和唯一索引的情况
    def test_update_on_duplicate_frame_unique_argument_index(self):
        # GH#55509: GitHub issue标识号，用于跟踪相关问题
        # 创建一个DataFrame对象df，包含一个列"a"，值为[1, 1, 1]，索引为[1, 1, 2]，数据类型为intc
        df = DataFrame({"a": [1, 1, 1]}, index=[1, 1, 2], dtype=np.dtype("intc"))
        # 创建另一个DataFrame对象other，包含一个列"a"，值为[2, 3]，索引为[1, 2]，数据类型为intc
        other = DataFrame({"a": [2, 3]}, index=[1, 2], dtype=np.dtype("intc"))
        # 期望的结果DataFrame，用于与更新后的df进行比较
        expected = DataFrame({"a": [2, 2, 3]}, index=[1, 1, 2], dtype=np.dtype("intc"))
        # 使用update方法，将other中的数据合并到df中
        df.update(other)
        # 使用测试工具tm.assert_frame_equal比较df和期望的expected是否相同
        tm.assert_frame_equal(df, expected)
```