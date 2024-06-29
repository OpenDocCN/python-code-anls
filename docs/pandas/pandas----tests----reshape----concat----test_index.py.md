# `D:\src\scipysrc\pandas\pandas\tests\reshape\concat\test_index.py`

```
    # 导入深拷贝函数 deepcopy
    from copy import deepcopy

    # 导入 numpy 库，并使用 np 别名
    import numpy as np

    # 导入 pytest 库，用于测试
    import pytest

    # 导入 pandas 库，并从中导入 DataFrame, Index, MultiIndex, Series, concat 等
    import pandas as pd
    from pandas import (
        DataFrame,
        Index,
        MultiIndex,
        Series,
        concat,
    )

    # 导入 pandas 内部的测试模块
    import pandas._testing as tm

    # 定义一个测试类 TestIndexConcat
    class TestIndexConcat:

        # 定义测试方法 test_concat_ignore_index，接受一个名为 sort 的参数
        def test_concat_ignore_index(self, sort):
            # 创建 DataFrame frame1，包含三列数据
            frame1 = DataFrame(
                {"test1": ["a", "b", "c"], "test2": [1, 2, 3], "test3": [4.5, 3.2, 1.2]}
            )
            # 创建 DataFrame frame2，只包含一列数据
            frame2 = DataFrame({"test3": [5.2, 2.2, 4.3]})
            # 设置 frame1 的索引为 Index(["x", "y", "z"])
            frame1.index = Index(["x", "y", "z"])
            # 设置 frame2 的索引为 Index(["x", "y", "q"])
            frame2.index = Index(["x", "y", "q"])

            # 调用 concat 函数，沿着列方向连接 frame1 和 frame2，忽略索引，根据 sort 参数排序
            v1 = concat([frame1, frame2], axis=1, ignore_index=True, sort=sort)

            # 设置变量 nan 为 numpy 中的 NaN
            nan = np.nan
            # 创建预期结果的 DataFrame expected，包含四行数据
            expected = DataFrame(
                [
                    [nan, nan, nan, 4.3],
                    ["a", 1, 4.5, 5.2],
                    ["b", 2, 3.2, 2.2],
                    ["c", 3, 1.2, nan],
                ],
                index=Index(["q", "x", "y", "z"]),
            )
            # 如果 sort 参数为 False，则按照指定顺序选择 expected 的行
            if not sort:
                expected = expected.loc[["x", "y", "z", "q"]]

            # 使用测试模块中的 assert_frame_equal 函数比较 v1 和 expected，确保它们相等
            tm.assert_frame_equal(v1, expected)

        # 使用 pytest 的 parametrize 装饰器定义多组参数化测试
        @pytest.mark.parametrize(
            "name_in1,name_in2,name_in3,name_out",
            [
                ("idx", "idx", "idx", "idx"),
                ("idx", "idx", None, None),
                ("idx", None, None, None),
                ("idx1", "idx2", None, None),
                ("idx1", "idx1", "idx2", None),
                ("idx1", "idx2", "idx3", None),
                (None, None, None, None),
            ],
        )
        # 定义测试方法 test_concat_same_index_names，接受多个索引名称参数和一个输出索引名称参数
        def test_concat_same_index_names(self, name_in1, name_in2, name_in3, name_out):
            # 创建三个不同的索引对象
            indices = [
                Index(["a", "b", "c"], name=name_in1),
                Index(["b", "c", "d"], name=name_in2),
                Index(["c", "d", "e"], name=name_in3),
            ]
            # 根据每个索引创建对应的 DataFrame，列名称为 ["x", "y", "z"]
            frames = [
                DataFrame({c: [0, 1, 2]}, index=i) for i, c in zip(indices, ["x", "y", "z"])
            ]
            # 调用 concat 函数，沿着列方向连接 frames 中的 DataFrame
            result = concat(frames, axis=1)

            # 创建预期的索引对象 exp_ind，包含五个元素
            exp_ind = Index(["a", "b", "c", "d", "e"], name=name_out)
            # 创建预期结果的 DataFrame expected，包含三列数据
            expected = DataFrame(
                {
                    "x": [0, 1, 2, np.nan, np.nan],
                    "y": [np.nan, 0, 1, 2, np.nan],
                    "z": [np.nan, np.nan, 0, 1, 2],
                },
                index=exp_ind,
            )

            # 使用测试模块中的 assert_frame_equal 函数比较 result 和 expected，确保它们相等
            tm.assert_frame_equal(result, expected)

        # 定义测试方法 test_concat_rename_index，不接受任何参数
        def test_concat_rename_index(self):
            # 创建 DataFrame a，包含随机数据，列名称为 ["A", "B", "C"]，索引名为 "index_a"
            a = DataFrame(
                np.random.default_rng(2).random((3, 3)),
                columns=list("ABC"),
                index=Index(list("abc"), name="index_a"),
            )
            # 创建 DataFrame b，包含随机数据，列名称为 ["A", "B", "C"]，索引名为 "index_b"
            b = DataFrame(
                np.random.default_rng(2).random((3, 3)),
                columns=list("ABC"),
                index=Index(list("abc"), name="index_b"),
            )

            # 调用 concat 函数，沿着行方向连接 DataFrame a 和 b，使用 keys 和 names 参数进行重命名
            result = concat([a, b], keys=["key0", "key1"], names=["lvl0", "lvl1"])

            # 创建预期结果的 DataFrame exp，沿用 a 和 b 连接的结果，重命名索引名
            exp = concat([a, b], keys=["key0", "key1"], names=["lvl0"])
            names = list(exp.index.names)
            names[1] = "lvl1"
            exp.index.set_names(names, inplace=True)

            # 使用测试模块中的 assert_frame_equal 函数比较 result 和 exp，确保它们相等
            tm.assert_frame_equal(result, exp)
            # 使用 assert 语句检查 result 的索引名是否与 exp 的索引名相同
            assert result.index.names == exp.index.names
    # 测试拼接和复制索引的 Series
    def test_concat_copy_index_series(self, axis):
        # GH 29879：GitHub 上的 issue 编号
        ser = Series([1, 2])
        # 将两个 Series 沿指定轴拼接成一个新的 Series 或 DataFrame
        comb = concat([ser, ser], axis=axis)
        # 如果 axis 是 0 或 "index"，则断言新对象的索引与原始 Series 的索引不同
        if axis in [0, "index"]:
            assert comb.index is not ser.index
        else:
            # 否则断言新对象的索引与原始 Series 的索引相同
            assert comb.index is ser.index

    # 测试拼接和复制索引的 DataFrame
    def test_concat_copy_index_frame(self, axis):
        # GH 29879：GitHub 上的 issue 编号
        df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
        # 将两个 DataFrame 沿指定轴拼接成一个新的 DataFrame
        comb = concat([df, df], axis=axis)
        # 如果 axis 是 0 或 "index"，则断言新对象的索引与原始 DataFrame 的索引不同，
        # 并且断言新对象的列与原始 DataFrame 的列相同
        if axis in [0, "index"]:
            assert not comb.index.is_(df.index)
            assert comb.columns.is_(df.columns)
        # 如果 axis 是 1 或 "columns"，则断言新对象的索引与原始 DataFrame 的索引相同，
        # 并且断言新对象的列与原始 DataFrame 的列不同
        elif axis in [1, "columns"]:
            assert comb.index.is_(df.index)
            assert not comb.columns.is_(df.columns)

    # 测试默认索引的情况
    def test_default_index(self):
        # 测试情况：is_series 和 ignore_index
        s1 = Series([1, 2, 3], name="x")
        s2 = Series([4, 5, 6], name="y")
        # 沿指定轴拼接两个 Series，忽略原始索引，返回新的 DataFrame
        res = concat([s1, s2], axis=1, ignore_index=True)
        # 断言结果的列是默认的 RangeIndex
        assert isinstance(res.columns, pd.RangeIndex)
        exp = DataFrame([[1, 4], [2, 5], [3, 6]])
        # 使用 check_index_type=True 检查结果是否有 RangeIndex（默认索引）
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

        # 测试情况：is_series 和 所有输入都没有名称
        s1 = Series([1, 2, 3])
        s2 = Series([4, 5, 6])
        # 沿指定轴拼接两个 Series，保留原始索引，返回新的 DataFrame
        res = concat([s1, s2], axis=1, ignore_index=False)
        # 断言结果的列是默认的 RangeIndex
        assert isinstance(res.columns, pd.RangeIndex)
        exp = DataFrame([[1, 4], [2, 5], [3, 6]])
        exp.columns = pd.RangeIndex(2)
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

        # 测试情况：is_dataframe 和 ignore_index
        df1 = DataFrame({"A": [1, 2], "B": [5, 6]})
        df2 = DataFrame({"A": [3, 4], "B": [7, 8]})

        # 沿指定轴拼接两个 DataFrame，忽略原始索引，返回新的 DataFrame
        res = concat([df1, df2], axis=0, ignore_index=True)
        exp = DataFrame([[1, 5], [2, 6], [3, 7], [4, 8]], columns=["A", "B"])
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

        # 沿指定轴拼接两个 DataFrame，忽略原始索引，返回新的 DataFrame
        res = concat([df1, df2], axis=1, ignore_index=True)
        exp = DataFrame([[1, 5, 3, 7], [2, 6, 4, 8]])
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)
    def test_dups_index(self):
        # 定义一个测试函数，用于测试数据框处理重复索引的情况
        # GH 4771

        # 创建一个包含单一数据类型的数据框
        df = DataFrame(
            np.random.default_rng(2).integers(0, 10, size=40).reshape(10, 4),
            columns=["A", "A", "C", "C"],  # 设置列名，包含重复的列名
        )

        # 在水平方向拼接两次相同的数据框
        result = concat([df, df], axis=1)
        # 验证拼接后的前四列与原数据框相等
        tm.assert_frame_equal(result.iloc[:, :4], df)
        # 验证拼接后的后四列与原数据框相等
        tm.assert_frame_equal(result.iloc[:, 4:], df)

        # 在垂直方向拼接两次相同的数据框
        result = concat([df, df], axis=0)
        # 验证拼接后的前十行与原数据框相等
        tm.assert_frame_equal(result.iloc[:10], df)
        # 验证拼接后的后十行与原数据框相等
        tm.assert_frame_equal(result.iloc[10:], df)

        # 创建包含多种数据类型的数据框
        df = concat(
            [
                DataFrame(
                    np.random.default_rng(2).standard_normal((10, 4)),
                    columns=["A", "A", "B", "B"],  # 设置列名，包含重复的列名
                ),
                DataFrame(
                    np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2),
                    columns=["A", "C"],  # 设置不同的列名
                ),
            ],
            axis=1,
        )

        # 在水平方向拼接两次相同的数据框
        result = concat([df, df], axis=1)
        # 验证拼接后的前六列与原数据框相等
        tm.assert_frame_equal(result.iloc[:, :6], df)
        # 验证拼接后的后六列与原数据框相等
        tm.assert_frame_equal(result.iloc[:, 6:], df)

        # 在垂直方向拼接两次相同的数据框
        result = concat([df, df], axis=0)
        # 验证拼接后的前十行与原数据框相等
        tm.assert_frame_equal(result.iloc[:10], df)
        # 验证拼接后的后十行与原数据框相等
        tm.assert_frame_equal(result.iloc[10:], df)

        # 使用 _append 方法进行数据框的追加操作
        result = df.iloc[0:8, :]._append(df.iloc[8:])
        # 验证追加后的结果与原数据框相等
        tm.assert_frame_equal(result, df)

        # 连续使用 _append 方法进行数据框的连续追加操作
        result = df.iloc[0:8, :]._append(df.iloc[8:9])._append(df.iloc[9:10])
        # 验证连续追加后的结果与原数据框相等
        tm.assert_frame_equal(result, df)

        # 创建预期的拼接结果
        expected = concat([df, df], axis=0)
        # 使用 _append 方法进行数据框的拼接操作
        result = df._append(df)
        # 验证拼接后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)
class TestMultiIndexConcat:
    # 测试多重索引的拼接功能，使用随机数据的数据框
    def test_concat_multiindex_with_keys(self, multiindex_dataframe_random_data):
        # 获取随机数据的数据框
        frame = multiindex_dataframe_random_data
        # 获取数据框的索引
        index = frame.index
        # 使用 keys=[0, 1] 拼接数据框 frame 自身，设置拼接后的索引名为 "iteration"
        result = concat([frame, frame], keys=[0, 1], names=["iteration"])

        # 断言拼接后结果的索引名应为 ("iteration",) + frame 的索引名
        assert result.index.names == ("iteration",) + index.names
        # 断言拼接后结果中 key=0 对应的部分应与 frame 相等
        tm.assert_frame_equal(result.loc[0], frame)
        # 断言拼接后结果中 key=1 对应的部分应与 frame 相等
        tm.assert_frame_equal(result.loc[1], frame)
        # 断言拼接后结果的索引级数应为 3
        assert result.index.nlevels == 3

    # 测试在索引名称中包含 None 的情况
    def test_concat_multiindex_with_none_in_index_names(self):
        # GH 15787
        # 创建一个 MultiIndex，包含一个 level1 和一个无名的维度
        index = MultiIndex.from_product([[1], range(5)], names=["level1", None])
        # 创建一个数据框 df，指定列 "col" 和上面创建的 MultiIndex
        df = DataFrame({"col": range(5)}, index=index, dtype=np.int32)

        # 使用 keys=[1, 2] 拼接数据框 df 自身，设置拼接后的索引名为 "level2"
        result = concat([df, df], keys=[1, 2], names=["level2"])
        # 构建预期结果 expected，确保拼接后的 MultiIndex 结构符合预期
        index = MultiIndex.from_product(
            [[1, 2], [1], range(5)], names=["level2", "level1", None]
        )
        expected = DataFrame({"col": list(range(5)) * 2}, index=index, dtype=np.int32)
        tm.assert_frame_equal(result, expected)

        # 再次拼接数据框 df 和 df[:2]，使用 keys=[1, 2]，设置拼接后的索引名为 "level2"
        result = concat([df, df[:2]], keys=[1, 2], names=["level2"])
        # 构建预期结果 expected，确保拼接后的 MultiIndex 结构符合预期
        level2 = [1] * 5 + [2] * 2
        level1 = [1] * 7
        no_name = list(range(5)) + list(range(2))
        tuples = list(zip(level2, level1, no_name))
        index = MultiIndex.from_tuples(tuples, names=["level2", "level1", None])
        expected = DataFrame({"col": no_name}, index=index, dtype=np.int32)
        tm.assert_frame_equal(result, expected)

    # 测试当多重索引的级别为 RangeIndex 对象时的情况
    def test_concat_multiindex_rangeindex(self):
        # GH13542
        # 当多重索引级别为 RangeIndex 对象时，concat 存在一个长度为 1 的对象的 bug

        # 创建一个包含随机标准正态分布数据的数据框 df
        df = DataFrame(np.random.default_rng(2).standard_normal((9, 2)))
        # 将数据框 df 的索引设定为包含 RangeIndex 对象的 MultiIndex
        df.index = MultiIndex(
            levels=[pd.RangeIndex(3), pd.RangeIndex(3)],
            codes=[np.repeat(np.arange(3), 3), np.tile(np.arange(3), 3)],
        )

        # 对 df 的子集进行拼接，保留部分包含特定行的数据
        res = concat([df.iloc[[2, 3, 4], :], df.iloc[[5], :]])
        # 构建预期结果 exp，确保拼接后的数据与预期相符
        exp = df.iloc[[2, 3, 4, 5], :]
        tm.assert_frame_equal(res, exp)

    # 测试拼接包含深拷贝的多重索引数据框的情况
    def test_concat_multiindex_dfs_with_deepcopy(self):
        # GH 9967
        # 创建一个包含单个元素 MultiIndex 的示例 example_multiindex1
        example_multiindex1 = MultiIndex.from_product([["a"], ["b"]])
        # 创建一个数据框 example_dataframe1，包含一列数据，使用上述 MultiIndex
        example_dataframe1 = DataFrame([0], index=example_multiindex1)

        # 创建另一个包含单个元素 MultiIndex 的示例 example_multiindex2
        example_multiindex2 = MultiIndex.from_product([["a"], ["c"]])
        # 创建一个数据框 example_dataframe2，包含一列数据，使用上述 MultiIndex
        example_dataframe2 = DataFrame([1], index=example_multiindex2)

        # 创建一个字典 example_dict，包含两个数据框示例
        example_dict = {"s1": example_dataframe1, "s2": example_dataframe2}
        # 创建预期结果的索引 expected_index，包含三个级别，分别命名为 "testname", None, None
        expected_index = MultiIndex(
            levels=[["s1", "s2"], ["a"], ["b", "c"]],
            codes=[[0, 1], [0, 0], [0, 1]],
            names=["testname", None, None],
        )
        # 创建预期结果 expected，确保拼接后的数据框与预期相符
        expected = DataFrame([[0], [1]], index=expected_index)

        # 使用深拷贝拼接 example_dict，设置拼接后的索引名为 "testname"
        result_copy = concat(deepcopy(example_dict), names=["testname"])
        tm.assert_frame_equal(result_copy, expected)
        # 直接拼接 example_dict，设置拼接后的索引名为 "testname"
        result_no_copy = concat(example_dict, names=["testname"])
        tm.assert_frame_equal(result_no_copy, expected)
    @pytest.mark.parametrize(
        "mi1_list",
        [
            [["a"], range(2)],  # 参数化测试数据 mi1_list 包含 [["a"], range(2)]
            [["b"], np.arange(2.0, 4.0)],  # 参数化测试数据 mi1_list 包含 [["b"], np.arange(2.0, 4.0)]
            [["c"], ["A", "B"]],  # 参数化测试数据 mi1_list 包含 [["c"], ["A", "B"]]
            [["d"], pd.date_range(start="2017", end="2018", periods=2)],  # 参数化测试数据 mi1_list 包含 [["d"], pd.date_range(start="2017", end="2018", periods=2)]
        ],
    )
    @pytest.mark.parametrize(
        "mi2_list",
        [
            [["a"], range(2)],  # 参数化测试数据 mi2_list 包含 [["a"], range(2)]
            [["b"], np.arange(2.0, 4.0)],  # 参数化测试数据 mi2_list 包含 [["b"], np.arange(2.0, 4.0)]
            [["c"], ["A", "B"]],  # 参数化测试数据 mi2_list 包含 [["c"], ["A", "B"]]
            [["d"], pd.date_range(start="2017", end="2018", periods=2)],  # 参数化测试数据 mi2_list 包含 [["d"], pd.date_range(start="2017", end="2018", periods=2)]
        ],
    )
    def test_concat_with_various_multiindex_dtypes(
        self, mi1_list: list, mi2_list: list
    ):
        # GitHub #23478
        # 使用 mi1_list 和 mi2_list 创建 MultiIndex 对象 mi1 和 mi2
        mi1 = MultiIndex.from_product(mi1_list)
        mi2 = MultiIndex.from_product(mi2_list)

        # 创建 DataFrame df1 和 df2，用零填充，列使用 mi1 和 mi2
        df1 = DataFrame(np.zeros((1, len(mi1))), columns=mi1)
        df2 = DataFrame(np.zeros((1, len(mi2))), columns=mi2)

        # 根据 mi1_list 和 mi2_list 的第一个元素比较，确定 expected_mi
        if mi1_list[0] == mi2_list[0]:
            expected_mi = MultiIndex(
                levels=[mi1_list[0], list(mi1_list[1])],
                codes=[[0, 0, 0, 0], [0, 1, 0, 1]],
            )
        else:
            expected_mi = MultiIndex(
                levels=[
                    mi1_list[0] + mi2_list[0],
                    list(mi1_list[1]) + list(mi2_list[1]),
                ],
                codes=[[0, 0, 1, 1], [0, 1, 2, 3]],
            )

        # 根据 expected_mi 创建预期的 DataFrame expected_df
        expected_df = DataFrame(np.zeros((1, len(expected_mi))), columns=expected_mi)

        # 使用 concat 合并 df1 和 df2，axis=1 表示按列合并，存储在 result_df 中
        with tm.assert_produces_warning(None):
            result_df = concat((df1, df2), axis=1)

        # 使用 assert_frame_equal 检查 result_df 是否与 expected_df 相等
        tm.assert_frame_equal(expected_df, result_df)

    def test_concat_multiindex_(self):
        # GitHub #44786
        # 创建包含单列和索引的 DataFrame df
        df = DataFrame({"col": ["a", "b", "c"]}, index=["1", "2", "2"])
        # 使用 concat 将 df 列表化，并使用 keys=["X"] 作为索引键
        df = concat([df], keys=["X"])

        # 创建二维可迭代对象 iterables
        iterables = [["X"], ["1", "2", "2"]]
        # 获取 df 的索引，存储在 result_index 中
        result_index = df.index
        # 根据 iterables 创建预期的索引 MultiIndex expected_index
        expected_index = MultiIndex.from_product(iterables)

        # 使用 assert_index_equal 检查 result_index 是否与 expected_index 相等
        tm.assert_index_equal(result_index, expected_index)

        # 将 df 存储在 result_df 中
        result_df = df
        # 根据 iterables 创建预期的 DataFrame expected_df
        expected_df = DataFrame(
            {"col": ["a", "b", "c"]}, index=MultiIndex.from_product(iterables)
        )
        # 使用 assert_frame_equal 检查 result_df 是否与 expected_df 相等
        tm.assert_frame_equal(result_df, expected_df)
    # 测试在拼接中当关键字不唯一时的行为
    def test_concat_with_key_not_unique(self, performance_warning):
        # 创建三个数据帧，每个包含一个名为"name"的列，内容分别为1, 2, 3
        df1 = DataFrame({"name": [1]})
        df2 = DataFrame({"name": [2]})
        df3 = DataFrame({"name": [3]})
        
        # 将三个数据帧按指定关键字拼接成一个新的数据帧df_a
        df_a = concat([df1, df2, df3], keys=["x", "y", "x"])
        
        # 由于索引未排序的多重索引导致警告
        with tm.assert_produces_warning(
            performance_warning, match="indexing past lexsort depth"
        ):
            # 选择df_a中的特定行，由于警告而触发
            out_a = df_a.loc[("x", 0), :]

        # 创建另一个数据帧df_b，包含"name"列和多重索引
        df_b = DataFrame(
            {"name": [1, 2, 3]}, index=Index([("x", 0), ("y", 0), ("x", 0)])
        )
        
        # 再次因为警告选择特定行
        with tm.assert_produces_warning(
            performance_warning, match="indexing past lexsort depth"
        ):
            out_b = df_b.loc[("x", 0)]

        # 断言两个输出的数据帧相等
        tm.assert_frame_equal(out_a, out_b)

        # 创建具有重复级别的数据帧，并期望引发错误
        df1 = DataFrame({"name": ["a", "a", "b"]})
        df2 = DataFrame({"name": ["a", "b"]})
        df3 = DataFrame({"name": ["c", "d"]})
        
        # 将这三个数据帧按指定关键字拼接成新的数据帧df_a
        df_a = concat([df1, df2, df3], keys=["x", "y", "x"])
        
        # 选择df_a中的特定行，由于警告而触发
        with tm.assert_produces_warning(
            performance_warning, match="indexing past lexsort depth"
        ):
            out_a = df_a.loc[("x", 0), :]

        # 创建另一个数据帧df_b，包含多重索引和"name"列
        df_b = DataFrame(
            {
                "a": ["x", "x", "x", "y", "y", "x", "x"],
                "b": [0, 1, 2, 0, 1, 0, 1],
                "name": list("aababcd"),
            }
        ).set_index(["a", "b"])
        
        # 由于警告选择特定行
        with tm.assert_produces_warning(
            performance_warning, match="indexing past lexsort depth"
        ):
            out_b = df_b.loc[("x", 0), :]

        # 断言两个输出的数据帧相等
        tm.assert_frame_equal(out_a, out_b)

    # 测试在拼接中使用重复级别时的行为
    def test_concat_with_duplicated_levels(self):
        # 关键字levels应该是唯一的
        df1 = DataFrame({"A": [1]}, index=["x"])
        df2 = DataFrame({"A": [1]}, index=["y"])
        msg = r"Level values not unique: \['x', 'y', 'y'\]"
        
        # 预期拼接这两个数据帧会引发值错误，并匹配特定消息
        with pytest.raises(ValueError, match=msg):
            concat([df1, df2], keys=["x", "y"], levels=[["x", "y", "y"]])

    # 使用不同的levels参数进行参数化测试，测试在拼接中使用levels和无键时的行为
    @pytest.mark.parametrize("levels", [[["x", "y"]], [["x", "y", "y"]]])
    def test_concat_with_levels_with_none_keys(self, levels):
        df1 = DataFrame({"A": [1]}, index=["x"])
        df2 = DataFrame({"A": [1]}, index=["y"])
        msg = "levels supported only when keys is not None"
        
        # 预期拼接这两个数据帧会引发值错误，并匹配特定消息
        with pytest.raises(ValueError, match=msg):
            concat([df1, df2], levels=levels)

    # 测试拼接结果中的RangeIndex行为
    def test_concat_range_index_result(self):
        # GitHub issue #47501
        df1 = DataFrame({"a": [1, 2]})
        df2 = DataFrame({"b": [1, 2]})

        # 将两个数据帧按列拼接成新的数据帧result，使用排序并且指定轴为1
        result = concat([df1, df2], sort=True, axis=1)
        
        # 期望的预期输出数据帧
        expected = DataFrame({"a": [1, 2], "b": [1, 2]})
        
        # 断言拼接结果与预期结果相等
        tm.assert_frame_equal(result, expected)
        
        # 期望的索引应为RangeIndex(0, 2)
        expected_index = pd.RangeIndex(0, 2)
        
        # 断言拼接结果的索引与预期索引完全相等
        tm.assert_index_equal(result.index, expected_index, exact=True)
    def test_concat_index_keep_dtype(self):
        # 定义一个测试函数，用于测试数据帧的连接操作，保持列的数据类型
        # GH#47329 表示GitHub上的issue编号
        df1 = DataFrame([[0, 1, 1]], columns=Index([1, 2, 3], dtype="object"))
        # 创建第一个数据帧，包含一个列表，并指定列索引，数据类型为对象
        df2 = DataFrame([[0, 1]], columns=Index([1, 2], dtype="object"))
        # 创建第二个数据帧，包含一个列表，并指定列索引，数据类型为对象
        result = concat([df1, df2], ignore_index=True, join="outer", sort=True)
        # 进行数据帧的连接操作，忽略索引，使用外连接，排序结果
        expected = DataFrame(
            [[0, 1, 1.0], [0, 1, np.nan]], columns=Index([1, 2, 3], dtype="object")
        )
        # 创建预期的数据帧，包含两行数据，预期结果中的列索引数据类型为对象
        tm.assert_frame_equal(result, expected)
        # 使用测试框架的函数来比较实际结果和预期结果的数据帧

    def test_concat_index_keep_dtype_ea_numeric(self, any_numeric_ea_dtype):
        # 定义一个测试函数，测试连接数据帧并保持列的数据类型，其中列的数据类型是可选的
        # GH#47329 表示GitHub上的issue编号
        df1 = DataFrame(
            [[0, 1, 1]], columns=Index([1, 2, 3], dtype=any_numeric_ea_dtype)
        )
        # 创建第一个数据帧，包含一个列表，并指定列索引和可选的数值型数据类型
        df2 = DataFrame([[0, 1]], columns=Index([1, 2], dtype=any_numeric_ea_dtype))
        # 创建第二个数据帧，包含一个列表，并指定列索引和可选的数值型数据类型
        result = concat([df1, df2], ignore_index=True, join="outer", sort=True)
        # 进行数据帧的连接操作，忽略索引，使用外连接，排序结果
        expected = DataFrame(
            [[0, 1, 1.0], [0, 1, np.nan]],
            columns=Index([1, 2, 3], dtype=any_numeric_ea_dtype),
        )
        # 创建预期的数据帧，包含两行数据，预期结果中的列索引使用可选的数值型数据类型
        tm.assert_frame_equal(result, expected)
        # 使用测试框架的函数来比较实际结果和预期结果的数据帧

    @pytest.mark.parametrize("dtype", ["Int8", "Int16", "Int32"])
    def test_concat_index_find_common(self, dtype):
        # 定义一个测试函数，测试连接数据帧并查找共同列，其中列的数据类型是指定的
        # GH#47329 表示GitHub上的issue编号
        df1 = DataFrame([[0, 1, 1]], columns=Index([1, 2, 3], dtype=dtype))
        # 创建第一个数据帧，包含一个列表，并指定列索引和指定的整数型数据类型
        df2 = DataFrame([[0, 1]], columns=Index([1, 2], dtype="Int32"))
        # 创建第二个数据帧，包含一个列表，并指定列索引和整数型数据类型为"Int32"
        result = concat([df1, df2], ignore_index=True, join="outer", sort=True)
        # 进行数据帧的连接操作，忽略索引，使用外连接，排序结果
        expected = DataFrame(
            [[0, 1, 1.0], [0, 1, np.nan]], columns=Index([1, 2, 3], dtype="Int32")
        )
        # 创建预期的数据帧，包含两行数据，预期结果中的列索引使用整数型数据类型"Int32"
        tm.assert_frame_equal(result, expected)
        # 使用测试框架的函数来比较实际结果和预期结果的数据帧

    def test_concat_axis_1_sort_false_rangeindex(self, using_infer_string):
        # 定义一个测试函数，测试连接序列并在轴1上进行排序，使用范围索引
        # GH 46675 表示GitHub上的issue编号
        s1 = Series(["a", "b", "c"])
        # 创建第一个序列，包含字符串列表
        s2 = Series(["a", "b"])
        # 创建第二个序列，包含字符串列表
        s3 = Series(["a", "b", "c", "d"])
        # 创建第三个序列，包含字符串列表
        s4 = Series(
            [], dtype=object if not using_infer_string else "string[pyarrow_numpy]"
        )
        # 创建第四个序列，根据条件设置数据类型为对象或指定的推断字符串类型
        result = concat(
            [s1, s2, s3, s4], sort=False, join="outer", ignore_index=False, axis=1
        )
        # 进行序列的连接操作，不排序，使用外连接，不忽略索引，操作轴为1
        expected = DataFrame(
            [
                ["a"] * 3 + [np.nan],
                ["b"] * 3 + [np.nan],
                ["c", np.nan] * 2,
                [np.nan] * 2 + ["d"] + [np.nan],
            ],
            dtype=object if not using_infer_string else "string[pyarrow_numpy]",
        )
        # 创建预期的数据帧，包含四行数据，根据条件设置列数据类型为对象或指定的推断字符串类型
        tm.assert_frame_equal(
            result, expected, check_index_type=True, check_column_type=True
        )
        # 使用测试框架的函数来比较实际结果和预期结果的数据帧
```