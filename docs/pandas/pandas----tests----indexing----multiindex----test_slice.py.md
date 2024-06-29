# `D:\src\scipysrc\pandas\pandas\tests\indexing\multiindex\test_slice.py`

```
    # 导入所需的模块和库
    from datetime import (
        datetime,  # 导入datetime类，用于处理日期和时间
        timedelta,  # 导入timedelta类，用于处理时间间隔
    )

    import numpy as np  # 导入NumPy库，用于数值计算
    import pytest  # 导入pytest模块，用于编写和运行测试

    from pandas.errors import UnsortedIndexError  # 从pandas.errors模块中导入UnsortedIndexError类

    import pandas as pd  # 导入pandas库，用于数据分析和处理
    from pandas import (
        DataFrame,  # 导入DataFrame类，用于处理表格数据
        Index,  # 导入Index类，用于处理索引
        MultiIndex,  # 导入MultiIndex类，用于处理多层索引
        Series,  # 导入Series类，用于处理一维数据
        Timestamp,  # 导入Timestamp类，用于处理时间戳
    )
    import pandas._testing as tm  # 导入pandas._testing模块，用于测试辅助工具
    from pandas.tests.indexing.common import _mklbl  # 从pandas.tests.indexing.common模块中导入_mklbl函数


class TestMultiIndexSlicers:
    def test_multiindex_slicers_non_unique(self):
        # GH 7106
        # 非唯一多层索引支持
        # 创建包含多层索引的DataFrame，并按索引排序
        df = (
            DataFrame(
                {
                    "A": ["foo", "foo", "foo", "foo"],
                    "B": ["a", "a", "a", "a"],
                    "C": [1, 2, 1, 3],
                    "D": [1, 2, 3, 4],
                }
            )
            .set_index(["A", "B", "C"])  # 将"A", "B", "C"列设为索引
            .sort_index()  # 按索引排序
        )
        assert not df.index.is_unique  # 断言索引不是唯一的
        # 创建预期结果的DataFrame，并设置相同的索引和排序
        expected = (
            DataFrame({"A": ["foo", "foo"], "B": ["a", "a"], "C": [1, 1], "D": [1, 3]})
            .set_index(["A", "B", "C"])
            .sort_index()
        )
        # 使用.loc方法和切片操作选取符合条件的子DataFrame，并断言结果与预期相等
        result = df.loc[(slice(None), slice(None), 1), :]
        tm.assert_frame_equal(result, expected)

        # 这相当于一个xs表达式（跨层级选择）
        result = df.xs(1, level=2, drop_level=False)
        tm.assert_frame_equal(result, expected)

        # 创建另一个包含非唯一多层索引的DataFrame，并按索引排序
        df = (
            DataFrame(
                {
                    "A": ["foo", "foo", "foo", "foo"],
                    "B": ["a", "a", "a", "a"],
                    "C": [1, 2, 1, 2],
                    "D": [1, 2, 3, 4],
                }
            )
            .set_index(["A", "B", "C"])  # 将"A", "B", "C"列设为索引
            .sort_index()  # 按索引排序
        )
        assert not df.index.is_unique  # 断言索引不是唯一的
        # 创建预期结果的DataFrame，并设置相同的索引和排序
        expected = (
            DataFrame({"A": ["foo", "foo"], "B": ["a", "a"], "C": [1, 1], "D": [1, 3]})
            .set_index(["A", "B", "C"])
            .sort_index()
        )
        # 使用.loc方法和切片操作选取符合条件的子DataFrame，并断言结果与预期相等
        result = df.loc[(slice(None), slice(None), 1), :]
        assert not result.index.is_unique  # 断言结果的索引不是唯一的
        tm.assert_frame_equal(result, expected)

        # GH12896
        # NumPy实现依赖性错误
        # 创建一个整数列表
        ints = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            12,
            13,
            14,
            14,
            16,
            17,
            18,
            19,
            200000,
            200000,
        ]
        n = len(ints)  # 计算列表的长度
        idx = MultiIndex.from_arrays([["a"] * n, ints])  # 使用数组创建多层索引
        result = Series([1] * n, index=idx)  # 创建一个Series对象，使用多层索引
        result = result.sort_index()  # 按索引排序
        result = result.loc[(slice(None), slice(100000))]  # 使用.loc方法和切片操作选取符合条件的子Series
        expected = Series([1] * (n - 2), index=idx[:-2]).sort_index()  # 创建预期结果的Series，并按索引排序
        tm.assert_series_equal(result, expected)  # 断言结果与预期相等
    # 定义测试函数，用于测试处理多级索引和时间日期类对象的切片操作
    def test_multiindex_slicers_datetimelike(self):
        # GH 7429：GitHub 上的 issue 编号，指明了这段代码是为了解决该问题而编写的
        # 当使用类似于日期时间的对象进行切片时，存在错误或不一致的行为

        # 创建一个日期时间列表，从特定日期开始，每隔一天增加一天，共生成6个日期
        dates = [datetime(2012, 1, 1, 12, 12, 12) + timedelta(days=i) for i in range(6)]
        # 创建一个频率列表
        freq = [1, 2]
        # 使用 MultiIndex 的 from_product 方法，生成日期和频率的笛卡尔积作为索引
        index = MultiIndex.from_product([dates, freq], names=["date", "frequency"])

        # 创建一个 DataFrame 对象
        df = DataFrame(
            np.arange(6 * 2 * 4, dtype="int64").reshape(-1, 4),
            index=index,
            columns=list("ABCD"),
        )

        # multi-axis slicing
        # 使用 pd.IndexSlice 创建 idx 对象，用于多级索引的切片操作
        idx = pd.IndexSlice

        # 以下开始进行多种切片操作，每次操作后与预期结果进行比较，确保结果正确性
        # 第一种切片方式，使用 Timestamp 对象进行切片
        expected = df.iloc[[0, 2, 4], [0, 1]]
        result = df.loc[
            (
                slice(
                    Timestamp("2012-01-01 12:12:12"), Timestamp("2012-01-03 12:12:12")
                ),
                slice(1, 1),
            ),
            slice("A", "B"),
        ]
        tm.assert_frame_equal(result, expected)

        # 第二种切片方式，使用 pd.IndexSlice 对象和 Timestamp 进行切片
        result = df.loc[
            (
                idx[
                    Timestamp("2012-01-01 12:12:12") : Timestamp("2012-01-03 12:12:12")
                ],
                idx[1:1],
            ),
            slice("A", "B"),
        ]
        tm.assert_frame_equal(result, expected)

        # 第三种切片方式，结合 Timestamp 和整数进行切片
        result = df.loc[
            (
                slice(
                    Timestamp("2012-01-01 12:12:12"), Timestamp("2012-01-03 12:12:12")
                ),
                1,
            ),
            slice("A", "B"),
        ]
        tm.assert_frame_equal(result, expected)

        # 使用字符串进行切片操作的示例
        result = df.loc[
            (slice("2012-01-01 12:12:12", "2012-01-03 12:12:12"), slice(1, 1)),
            slice("A", "B"),
        ]
        tm.assert_frame_equal(result, expected)

        # 结合字符串和 pd.IndexSlice 对象进行切片操作的示例
        result = df.loc[
            (idx["2012-01-01 12:12:12":"2012-01-03 12:12:12"], 1), idx["A", "B"]
        ]
        tm.assert_frame_equal(result, expected)
    def test_multiindex_slicers_edges(self):
        # 测试多级索引切片的边缘情况

        # 创建包含多个边缘情况的DataFrame
        df = DataFrame(
            {
                "A": ["A0"] * 5 + ["A1"] * 5 + ["A2"] * 5,
                "B": ["B0", "B0", "B1", "B1", "B2"] * 3,
                "DATE": [
                    "2013-06-11",
                    "2013-07-02",
                    "2013-07-09",
                    "2013-07-30",
                    "2013-08-06",
                    "2013-06-11",
                    "2013-07-02",
                    "2013-07-09",
                    "2013-07-30",
                    "2013-08-06",
                    "2013-09-03",
                    "2013-10-01",
                    "2013-07-09",
                    "2013-08-06",
                    "2013-09-03",
                ],
                "VALUES": [22, 35, 14, 9, 4, 40, 18, 4, 2, 5, 1, 2, 3, 4, 2],
            }
        )

        # 将"DATE"列转换为日期时间格式
        df["DATE"] = pd.to_datetime(df["DATE"])

        # 在多级索引 ("A", "B", "DATE") 上设置索引
        df1 = df.set_index(["A", "B", "DATE"])

        # 根据索引排序 DataFrame
        df1 = df1.sort_index()

        # A1 - 获取所有属于"A0"和"A1"下的值
        result = df1.loc[(slice("A1")), :]
        expected = df1.iloc[0:10]
        tm.assert_frame_equal(result, expected)

        # A2 - 获取从开头到"A2"下的所有值
        result = df1.loc[(slice("A2")), :]
        expected = df1
        tm.assert_frame_equal(result, expected)

        # A3 - 获取所有属于"B1"或"B2"下的值
        result = df1.loc[(slice(None), slice("B1", "B2")), :]
        expected = df1.iloc[[2, 3, 4, 7, 8, 9, 12, 13, 14]]
        tm.assert_frame_equal(result, expected)

        # A4 - 获取日期介于"2013-07-02"和"2013-07-09"之间的所有值
        result = df1.loc[(slice(None), slice(None), slice("20130702", "20130709")), :]
        expected = df1.iloc[[1, 2, 6, 7, 12]]
        tm.assert_frame_equal(result, expected)

        # B1 - 获取在"A2"下的"B0"中的所有值
        result = df1.loc[(slice("A2"), slice("B0")), :]
        expected = df1.iloc[[0, 1, 5, 6, 10, 11]]
        tm.assert_frame_equal(result, expected)

        # B2 - 获取所有在"B2"下的值（类似于A2中的操作）
        result = df1.loc[(slice(None), slice("B2")), :]
        expected = df1
        tm.assert_frame_equal(result, expected)

        # B3 - 获取从"B1"到"B2"并且日期不超过"2013-08-06"的所有值
        result = df1.loc[(slice(None), slice("B1", "B2"), slice("2013-08-06")), :]
        expected = df1.iloc[[2, 3, 4, 7, 8, 9, 12, 13]]
        tm.assert_frame_equal(result, expected)

        # B4 - 和A4类似，但日期范围的开始不是一个关键索引
        #      显示在部分选择切片上的索引
        result = df1.loc[(slice(None), slice(None), slice("20130701", "20130709")), :]
        expected = df1.iloc[[1, 2, 6, 7, 12]]
        tm.assert_frame_equal(result, expected)
    # 定义测试函数，用于测试多层索引切片的文档示例
    def test_per_axis_per_level_doc_examples(self):
        # 创建一个 pd.IndexSlice 对象用于简化多层索引的切片操作
        idx = pd.IndexSlice

        # 从 indexing.rst / advanced 中的示例创建 MultiIndex
        index = MultiIndex.from_product(
            [_mklbl("A", 4), _mklbl("B", 2), _mklbl("C", 4), _mklbl("D", 2)]
        )
        # 创建 MultiIndex 列
        columns = MultiIndex.from_tuples(
            [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")],
            names=["lvl0", "lvl1"],
        )
        # 创建 DataFrame，填充数据为整型，形状与 index 和 columns 对应
        df = DataFrame(
            np.arange(len(index) * len(columns), dtype="int64").reshape(
                (len(index), len(columns))
            ),
            index=index,
            columns=columns,
        )

        # 根据条件进行索引和预期结果比较，验证结果是否符合预期
        result = df.loc[(slice("A1", "A3"), slice(None), ["C1", "C3"]), :]
        expected = df.loc[
            [
                (
                    a,
                    b,
                    c,
                    d,
                )
                for a, b, c, d in df.index.values
                if a in ("A1", "A2", "A3") and c in ("C1", "C3")
            ]
        ]
        tm.assert_frame_equal(result, expected)

        # 使用 IndexSlice 进行切片操作，并验证结果是否符合预期
        result = df.loc[idx["A1":"A3", :, ["C1", "C3"]], :]
        tm.assert_frame_equal(result, expected)

        # 根据条件进行索引和预期结果比较，验证结果是否符合预期
        result = df.loc[(slice(None), slice(None), ["C1", "C3"]), :]
        expected = df.loc[
            [
                (
                    a,
                    b,
                    c,
                    d,
                )
                for a, b, c, d in df.index.values
                if c in ("C1", "C3")
            ]
        ]
        tm.assert_frame_equal(result, expected)

        # 使用 IndexSlice 进行切片操作，并验证结果是否符合预期
        result = df.loc[idx[:, :, ["C1", "C3"]], :]
        tm.assert_frame_equal(result, expected)

        # 对未排序的 MultiIndex 执行切片操作，预期引发 UnsortedIndexError 异常
        msg = (
            "MultiIndex slicing requires the index to be lexsorted: "
            r"slicing on levels \[1\], lexsort depth 1"
        )
        with pytest.raises(UnsortedIndexError, match=msg):
            df.loc["A1", ("a", slice("foo"))]

        # 对未排序的 MultiIndex 执行切片操作，预期结果与指定的列相匹配
        tm.assert_frame_equal(
            df.loc["A1", (slice(None), "foo")], df.loc["A1"].iloc[:, [0, 2]]
        )

        # 对 DataFrame 的列索引进行排序
        df = df.sort_index(axis=1)

        # 根据条件进行切片操作，并验证结果是否符合预期
        df.loc["A1", (slice(None), "foo")]
        df.loc[(slice(None), slice(None), ["C1", "C3"]), (slice(None), "foo")]

        # 使用 loc(axis=0) 进行切片赋值操作
        df.loc(axis=0)[:, :, ["C1", "C3"]] = -10
    # 测试使用 loc 方法对多级索引的 DataFrame 进行轴向（axis）参数的定位和选择

    def test_loc_axis_arguments(self):
        # 创建一个多级索引对象 index，包含四个级别，每个级别对应一组标签
        index = MultiIndex.from_product(
            [_mklbl("A", 4), _mklbl("B", 2), _mklbl("C", 4), _mklbl("D", 2)]
        )
        # 创建一个多级索引对象 columns，每个级别下有两个标签，同时指定级别的名称
        columns = MultiIndex.from_tuples(
            [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")],
            names=["lvl0", "lvl1"],
        )
        # 创建一个 DataFrame 对象 df，填充数据为整数序列，索引为 index，列为 columns，然后按索引和列排序
        df = (
            DataFrame(
                np.arange(len(index) * len(columns), dtype="int64").reshape(
                    (len(index), len(columns))
                ),
                index=index,
                columns=columns,
            )
            .sort_index()  # 按索引排序
            .sort_index(axis=1)  # 按列排序
        )

        # 对 axis 0 进行定位和选择
        result = df.loc(axis=0)["A1":"A3", :, ["C1", "C3"]]
        # 预期的结果是从 df 中选择满足条件的行索引
        expected = df.loc[
            [
                (
                    a,
                    b,
                    c,
                    d,
                )
                for a, b, c, d in df.index.values
                if a in ("A1", "A2", "A3") and c in ("C1", "C3")
            ]
        ]
        # 使用 pytest 的 assert_frame_equal 方法比较结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)

        # 对 axis 0 进行更简单的定位和选择，不指定参数名称
        result = df.loc(axis="index")[:, :, ["C1", "C3"]]
        # 预期的结果是从 df 中选择满足条件的行索引
        expected = df.loc[
            [
                (
                    a,
                    b,
                    c,
                    d,
                )
                for a, b, c, d in df.index.values
                if c in ("C1", "C3")
            ]
        ]
        # 使用 pytest 的 assert_frame_equal 方法比较结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)

        # 对 axis 1 进行定位和选择
        result = df.loc(axis=1)[:, "foo"]
        # 预期的结果是从 df 中选择满足条件的列索引
        expected = df.loc[:, (slice(None), "foo")]
        # 使用 pytest 的 assert_frame_equal 方法比较结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)

        # 对 axis 1 进行更简单的定位和选择，不指定参数名称
        result = df.loc(axis="columns")[:, "foo"]
        # 预期的结果是从 df 中选择满足条件的列索引
        expected = df.loc[:, (slice(None), "foo")]
        # 使用 pytest 的 assert_frame_equal 方法比较结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)

        # 处理无效的轴参数情况
        for i in [-1, 2, "foo"]:
            # 预期的错误消息
            msg = f"No axis named {i} for object type DataFrame"
            # 使用 pytest 的 assert_raises 方法验证是否会抛出 ValueError 异常，并且错误消息匹配预期
            with pytest.raises(ValueError, match=msg):
                df.loc(axis=i)[:, :, ["C1", "C3"]]

    def test_loc_axis_single_level_multi_col_indexing_multiindex_col_df(self):
        # GH29519
        # 创建一个多级索引的 DataFrame 对象 df
        df = DataFrame(
            np.arange(27).reshape(3, 9),
            columns=MultiIndex.from_product([["a1", "a2", "a3"], ["b1", "b2", "b3"]]),
        )
        # 对 axis 1 进行定位和选择
        result = df.loc(axis=1)["a1":"a2"]
        # 预期的结果是选择 df 的部分列，从左到右直到第一个标签是 "a2" 为止
        expected = df.iloc[:, :-3]
        # 使用 pytest 的 assert_frame_equal 方法比较结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)

    def test_loc_axis_single_level_single_col_indexing_multiindex_col_df(self):
        # GH29519
        # 创建一个多级索引的 DataFrame 对象 df
        df = DataFrame(
            np.arange(27).reshape(3, 9),
            columns=MultiIndex.from_product([["a1", "a2", "a3"], ["b1", "b2", "b3"]]),
        )
        # 对 axis 1 进行定位和选择
        result = df.loc(axis=1)["a1"]
        # 预期的结果是选择 df 的部分列，只选择第一个标签是 "a1" 的列，并且重新命名列名
        expected = df.iloc[:, :3]
        expected.columns = ["b1", "b2", "b3"]
        # 使用 pytest 的 assert_frame_equal 方法比较结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)
    # 定义测试函数：测试在单索引列数据框上进行单级别索引
    def test_loc_ax_single_level_indexer_simple_df(self):
        # GH29519: 对应 GitHub 问题编号
        # 测试在单索引列数据框上进行单级别索引
        df = DataFrame(np.arange(9).reshape(3, 3), columns=["a", "b", "c"])
        # 使用 loc(axis=1)["a"] 进行单级别索引，选取列名为 "a" 的数据列
        result = df.loc(axis=1)["a"]
        # 期望结果是一个 Series，包含 [0, 3, 6]，列名为 "a"
        expected = Series(np.array([0, 3, 6]), name="a")
        # 使用测试框架检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    # 定义测试函数：测试多级别标签切片，使用负步长
    def test_multiindex_label_slicing_with_negative_step(self):
        ser = Series(
            np.arange(20), MultiIndex.from_product([list("abcde"), np.arange(4)])
        )
        SLC = pd.IndexSlice

        # 使用测试框架检查 ser 使用 SLC[::-1] 切片后的结果是否等效于 SLC[::-1]
        tm.assert_indexing_slices_equivalent(ser, SLC[::-1], SLC[::-1])

        # 检查按标签 "d" 到开头使用负步长切片后的结果是否等效
        tm.assert_indexing_slices_equivalent(ser, SLC["d"::-1], SLC[15::-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[("d",) :: -1], SLC[15::-1])

        # 检查按标签从开头到 "d" 使用负步长切片后的结果是否等效
        tm.assert_indexing_slices_equivalent(ser, SLC[:"d":-1], SLC[:11:-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[: ("d",) : -1], SLC[:11:-1])

        # 检查按标签 "d" 到 "b" 使用负步长切片后的结果是否等效
        tm.assert_indexing_slices_equivalent(ser, SLC["d":"b":-1], SLC[15:3:-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[("d",) : "b" : -1], SLC[15:3:-1])
        tm.assert_indexing_slices_equivalent(ser, SLC["d" : ("b",) : -1], SLC[15:3:-1])
        tm.assert_indexing_slices_equivalent(
            ser, SLC[("d",) : ("b",) : -1], SLC[15:3:-1]
        )

        # 检查按标签 "b" 到 "d" 使用负步长切片后的结果是否等效
        tm.assert_indexing_slices_equivalent(ser, SLC["b":"d":-1], SLC[:0])

        # 检查按标签 ("c", 2) 到开头使用负步长切片后的结果是否等效
        tm.assert_indexing_slices_equivalent(ser, SLC[("c", 2) :: -1], SLC[10::-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[: ("c", 2) : -1], SLC[:9:-1])

        # 检查按标签 ("e", 0) 到 ("c", 2) 使用负步长切片后的结果是否等效
        tm.assert_indexing_slices_equivalent(
            ser, SLC[("e", 0) : ("c", 2) : -1], SLC[16:9:-1]
        )

    # 定义测试函数：测试多级别索引的第一级别切片
    def test_multiindex_slice_first_level(self):
        # GH 12697: 对应 GitHub 问题编号
        freq = ["a", "b", "c", "d"]
        idx = MultiIndex.from_product([freq, range(500)])
        df = DataFrame(list(range(2000)), index=idx, columns=["Test"])
        # 使用 loc[pd.IndexSlice[:, 30:70], :] 进行切片操作
        df_slice = df.loc[pd.IndexSlice[:, 30:70], :]
        # 检查切片后选择 "a" 标签的结果是否等于预期的 DataFrame
        result = df_slice.loc["a"]
        expected = DataFrame(list(range(30, 71)), columns=["Test"], index=range(30, 71))
        tm.assert_frame_equal(result, expected)
        # 检查切片后选择 "d" 标签的结果是否等于预期的 DataFrame
        result = df_slice.loc["d"]
        expected = DataFrame(
            list(range(1530, 1571)), columns=["Test"], index=range(30, 71)
        )
        tm.assert_frame_equal(result, expected)

    # 定义测试函数：测试整数索引的 Series 切片
    def test_int_series_slicing(self, multiindex_year_month_day_dataframe_random_data):
        ymd = multiindex_year_month_day_dataframe_random_data
        s = ymd["A"]
        # 检查从索引位置 5 开始的 Series 切片是否等于重新索引后的预期结果
        result = s[5:]
        expected = s.reindex(s.index[5:])
        tm.assert_series_equal(result, expected)

        # 创建 s 的副本进行修改
        s = ymd["A"].copy()
        exp = ymd["A"].copy()
        # 将索引位置从 5 开始的值设置为 0
        s[5:] = 0
        exp.iloc[5:] = 0
        # 检查修改后的 Series 值是否与预期相等
        tm.assert_numpy_array_equal(s.values, exp.values)

        # 检查从索引位置 5 开始的 DataFrame 切片是否等于重新索引后的预期结果
        result = ymd[5:]
        expected = ymd.reindex(s.index[5:])
        tm.assert_frame_equal(result, expected)
    @pytest.mark.parametrize(
        "dtype, loc, iloc",
        [
            # 参数化测试数据: 数据类型为int, 步长为-1
            ("int", slice(None, None, -1), slice(None, None, -1)),
            ("int", slice(3, None, -1), slice(3, None, -1)),
            ("int", slice(None, 1, -1), slice(None, 0, -1)),
            ("int", slice(3, 1, -1), slice(3, 0, -1)),
            # 参数化测试数据: 数据类型为int, 步长为-2
            ("int", slice(None, None, -2), slice(None, None, -2)),
            ("int", slice(3, None, -2), slice(3, None, -2)),
            ("int", slice(None, 1, -2), slice(None, 0, -2)),
            ("int", slice(3, 1, -2), slice(3, 0, -2)),
            # 参数化测试数据: 数据类型为str, 步长为-1
            ("str", slice(None, None, -1), slice(None, None, -1)),
            ("str", slice("d", None, -1), slice(3, None, -1)),
            ("str", slice(None, "b", -1), slice(None, 0, -1)),
            ("str", slice("d", "b", -1), slice(3, 0, -1)),
            # 参数化测试数据: 数据类型为str, 步长为-2
            ("str", slice(None, None, -2), slice(None, None, -2)),
            ("str", slice("d", None, -2), slice(3, None, -2)),
            ("str", slice(None, "b", -2), slice(None, 0, -2)),
            ("str", slice("d", "b", -2), slice(3, 0, -2)),
        ],
    )
    # 定义测试方法：测试负步长切片操作
    def test_loc_slice_negative_stepsize(self, dtype, loc, iloc):
        # GH#38071
        # 定义标签数据
        labels = {
            "str": list("abcde"),
            "int": range(5),
        }[dtype]

        # 创建一个包含两个相同标签数组的多级索引
        mi = MultiIndex.from_arrays([labels] * 2)
        # 创建一个DataFrame对象，索引为mi，列为["A"]，所有值为1.0
        df = DataFrame(1.0, index=mi, columns=["A"])

        # 定义索引切片对象
        SLC = pd.IndexSlice

        # 期望的iloc切片结果
        expected = df.iloc[iloc, :]
        # 使用loc切片进行索引，获取结果
        result_get_loc = df.loc[SLC[loc], :]
        # 使用多级索引的loc切片进行索引，获取结果
        result_get_locs_level_0 = df.loc[SLC[loc, :], :]
        # 使用第一级索引的loc切片进行索引，获取结果
        result_get_locs_level_1 = df.loc[SLC[:, loc], :]

        # 断言获取的loc切片结果与期望的iloc切片结果相等
        tm.assert_frame_equal(result_get_loc, expected)
        # 断言多级索引的loc切片结果与期望的iloc切片结果相等
        tm.assert_frame_equal(result_get_locs_level_0, expected)
        # 断言第一级索引的loc切片结果与期望的iloc切片结果相等
        tm.assert_frame_equal(result_get_locs_level_1, expected)
```