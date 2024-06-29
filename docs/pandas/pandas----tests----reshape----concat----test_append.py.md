# `D:\src\scipysrc\pandas\pandas\tests\reshape\concat\test_append.py`

```
import datetime as dt  # 导入datetime模块，并使用别名dt
from itertools import combinations  # 导入itertools模块中的combinations函数

import dateutil  # 导入dateutil模块
import numpy as np  # 导入numpy库，并使用别名np
import pytest  # 导入pytest库

import pandas as pd  # 导入pandas库，并使用别名pd
from pandas import (  # 从pandas库中导入多个函数和类
    DataFrame,
    Index,
    Series,
    Timestamp,
    concat,
    isna,
)
import pandas._testing as tm  # 导入pandas._testing模块，并使用别名tm


class TestAppend:
    def test_append(self, sort, float_frame):
        mixed_frame = float_frame.copy()  # 复制float_frame，得到mixed_frame
        mixed_frame["foo"] = "bar"  # 在mixed_frame中添加新列"foo"，并赋值为"bar"

        begin_index = float_frame.index[:5]  # 获取float_frame前5个索引
        end_index = float_frame.index[5:]  # 获取float_frame从第5个索引开始的后续索引

        begin_frame = float_frame.reindex(begin_index)  # 根据begin_index重新索引float_frame，得到begin_frame
        end_frame = float_frame.reindex(end_index)  # 根据end_index重新索引float_frame，得到end_frame

        appended = begin_frame._append(end_frame)  # 将begin_frame和end_frame合并，得到appended
        tm.assert_almost_equal(appended["A"], float_frame["A"])  # 断言appended中"A"列接近于float_frame中"A"列的值

        del end_frame["A"]  # 删除end_frame中的"A"列
        partial_appended = begin_frame._append(end_frame, sort=sort)  # 将begin_frame和修改后的end_frame按sort参数合并，得到partial_appended
        assert "A" in partial_appended  # 断言"partial_appended"包含"A"列

        partial_appended = end_frame._append(begin_frame, sort=sort)  # 将end_frame和begin_frame按sort参数合并，得到partial_appended
        assert "A" in partial_appended  # 断言"partial_appended"包含"A"列

        # mixed type handling
        appended = mixed_frame[:5]._append(mixed_frame[5:])  # 将mixed_frame的前5行和后续行合并，得到appended
        tm.assert_frame_equal(appended, mixed_frame)  # 断言appended与mixed_frame相等

        # what to test here
        mixed_appended = mixed_frame[:5]._append(float_frame[5:], sort=sort)  # 将mixed_frame的前5行与float_frame的后续行按sort参数合并，得到mixed_appended
        mixed_appended2 = float_frame[:5]._append(mixed_frame[5:], sort=sort)  # 将float_frame的前5行与mixed_frame的后续行按sort参数合并，得到mixed_appended2

        # all equal except 'foo' column
        tm.assert_frame_equal(
            mixed_appended.reindex(columns=["A", "B", "C", "D"]),  # 对mixed_appended按指定列重新索引
            mixed_appended2.reindex(columns=["A", "B", "C", "D"]),  # 对mixed_appended2按指定列重新索引
        )

    def test_append_empty(self, float_frame):
        empty = DataFrame()  # 创建一个空的DataFrame对象

        appended = float_frame._append(empty)  # 将float_frame与空的DataFrame对象empty合并，得到appended
        tm.assert_frame_equal(float_frame, appended)  # 断言appended与float_frame相等
        assert appended is not float_frame  # 断言appended不是float_frame本身

        appended = empty._append(float_frame)  # 将empty与float_frame合并，得到appended
        tm.assert_frame_equal(float_frame, appended)  # 断言appended与float_frame相等
        assert appended is not float_frame  # 断言appended不是float_frame本身

    def test_append_overlap_raises(self, float_frame):
        msg = "Indexes have overlapping values"  # 设置异常信息字符串
        with pytest.raises(ValueError, match=msg):  # 使用pytest断言捕获特定的ValueError异常，并匹配msg字符串
            float_frame._append(float_frame, verify_integrity=True)  # 尝试将float_frame与自身合并，同时验证完整性

    def test_append_new_columns(self):
        # see gh-6129: new columns
        df = DataFrame({"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}})  # 创建一个新的DataFrame对象df，包含两列"a"和"b"
        row = Series([5, 6, 7], index=["a", "b", "c"], name="z")  # 创建一个新的Series对象row，设置索引和名称
        expected = DataFrame(  # 创建预期的DataFrame对象expected
            {
                "a": {"x": 1, "y": 2, "z": 5},  # 第一列"a"的数据
                "b": {"x": 3, "y": 4, "z": 6},  # 第二列"b"的数据
                "c": {"z": 7},  # 第三列"c"的数据
            }
        )
        result = df._append(row)  # 将df与row合并，得到result
        tm.assert_frame_equal(result, expected)  # 断言result与expected相等

    def test_append_length0_frame(self, sort):
        df = DataFrame(columns=["A", "B", "C"])  # 创建一个指定列的空DataFrame对象df
        df3 = DataFrame(index=[0, 1], columns=["A", "B"])  # 创建一个指定索引和列的DataFrame对象df3
        df5 = df._append(df3, sort=sort)  # 将df与df3按sort参数合并，得到df5

        expected = DataFrame(index=[0, 1], columns=["A", "B", "C"])  # 创建预期的DataFrame对象expected
        tm.assert_frame_equal(df5, expected)  # 断言df5与expected相等
    # 测试函数：测试在DataFrame对象中追加记录的操作
    def test_append_records(self):
        # 创建一个包含两行数据的零数组，每行包含一个整数、一个浮点数和一个字符串
        arr1 = np.zeros((2,), dtype=("i4,f4,S10"))
        # 分别将两行数据赋值为 (1, 2.0, "Hello") 和 (2, 3.0, "World")
        arr1[:] = [(1, 2.0, "Hello"), (2, 3.0, "World")]

        # 创建一个包含三行数据的零数组，每行包含一个整数、一个浮点数和一个字符串
        arr2 = np.zeros((3,), dtype=("i4,f4,S10"))
        # 分别将三行数据赋值为 (3, 4.0, "foo"), (5, 6.0, "bar"), (7.0, 8.0, "baz")
        arr2[:] = [(3, 4.0, "foo"), (5, 6.0, "bar"), (7.0, 8.0, "baz")]

        # 使用数组 arr1 创建 DataFrame 对象 df1
        df1 = DataFrame(arr1)
        # 使用数组 arr2 创建 DataFrame 对象 df2
        df2 = DataFrame(arr2)

        # 调用 DataFrame 对象 df1 的 _append 方法，将 df2 追加到 df1 中，并忽略索引
        result = df1._append(df2, ignore_index=True)
        # 创建一个期望的 DataFrame 对象，包含 arr1 和 arr2 的拼接结果
        expected = DataFrame(np.concatenate((arr1, arr2)))
        # 断言 result 和 expected 在内容上相等
        tm.assert_frame_equal(result, expected)

    # 重写排序的 fixture，因为我们还想测试 sort 参数默认值为 None 的情况
    def test_append_sorts(self, sort):
        # 创建一个包含两列的 DataFrame 对象 df1
        df1 = DataFrame({"a": [1, 2], "b": [1, 2]}, columns=["b", "a"])
        # 创建一个包含两列的 DataFrame 对象 df2，并指定索引
        df2 = DataFrame({"a": [1, 2], "c": [3, 4]}, index=[2, 3])

        # 调用 DataFrame 对象 df1 的 _append 方法，将 df2 追加到 df1 中，根据 sort 参数进行排序
        result = df1._append(df2, sort=sort)

        # 根据 sort 参数的不同情况，创建不同的期望结果 DataFrame 对象
        # 当 sort 为 None 或 True 时，期望的 DataFrame 包含列 "a", "b", "c"
        expected = DataFrame(
            {"b": [1, 2, None, None], "a": [1, 2, 1, 2], "c": [None, None, 3, 4]},
            columns=["a", "b", "c"],
        )
        # 当 sort 为 False 时，调整期望的 DataFrame 列的顺序为 "b", "a", "c"
        if sort is False:
            expected = expected[["b", "a", "c"]]
        # 断言 result 和 expected 在内容上相等
        tm.assert_frame_equal(result, expected)

    # 测试函数：测试在具有不同列的 DataFrame 对象中进行追加操作
    def test_append_different_columns(self, sort):
        # 创建一个包含随机数据的 DataFrame 对象 df
        df = DataFrame(
            {
                "bools": np.random.default_rng(2).standard_normal(10) > 0,
                "ints": np.random.default_rng(2).integers(0, 10, 10),
                "floats": np.random.default_rng(2).standard_normal(10),
                "strings": ["foo", "bar"] * 5,
            }
        )

        # 从 df 中选取前5行，并且仅保留列 "bools", "ints", "floats"
        a = df[:5].loc[:, ["bools", "ints", "floats"]]
        # 从 df 中选取后5行，并且仅保留列 "strings", "ints", "floats"
        b = df[5:].loc[:, ["strings", "ints", "floats"]]

        # 调用 DataFrame 对象 a 的 _append 方法，将 b 追加到 a 中，并根据 sort 参数进行排序
        appended = a._append(b, sort=sort)
        # 断言前4行的 "strings" 列均为空值
        assert isna(appended["strings"][0:4]).all()
        # 断言后5行的 "bools" 列均为空值
        assert isna(appended["bools"][5:]).all()

    # 测试函数：测试在多个 DataFrame 对象中进行追加操作
    def test_append_many(self, sort, float_frame):
        # 将 float_frame 分成四个块
        chunks = [
            float_frame[:5],
            float_frame[5:10],
            float_frame[10:15],
            float_frame[15:],
        ]

        # 将 chunks[1:] 的所有块追加到 chunks[0] 中
        result = chunks[0]._append(chunks[1:])
        # 断言 result 和 float_frame 在内容上相等
        tm.assert_frame_equal(result, float_frame)

        # 复制 chunks[-1]，并向其添加名为 "foo" 的列
        chunks[-1] = chunks[-1].copy()
        chunks[-1]["foo"] = "bar"
        # 调用 chunks[0] 的 _append 方法，将 chunks[1:] 的所有块追加到 chunks[0] 中，并根据 sort 参数进行排序
        result = chunks[0]._append(chunks[1:], sort=sort)
        # 断言 result 的列只包含 float_frame 的列，并且 result 的 "foo" 列的后15行都为 "bar"
        tm.assert_frame_equal(result.loc[:, float_frame.columns], float_frame)
        assert (result["foo"][15:] == "bar").all()
        assert result["foo"][:15].isna().all()

    # 测试函数：测试在保留索引名的情况下进行追加操作
    def test_append_preserve_index_name(self):
        # 创建一个不含数据的 DataFrame 对象 df1，列名为 "A", "B", "C"，并将 "A" 列设为索引
        df1 = DataFrame(columns=["A", "B", "C"])
        df1 = df1.set_index(["A"])
        # 创建一个包含数据的 DataFrame 对象 df2，数据为 [[1, 4, 7], [2, 5, 8], [3, 6, 9]]，列名为 "A", "B", "C"，并将 "A" 列设为索引
        df2 = DataFrame(data=[[1, 4, 7], [2, 5, 8], [3, 6, 9]], columns=["A", "B", "C"])
        df2 = df2.set_index(["A"])

        # 调用 DataFrame 对象 df1 的 _append 方法，将 df2 追加到 df1 中
        result = df1._append(df2)
        # 断言 result 的索引名为 "A"
        assert result.index.name == "A"
    # 可以追加的索引对象列表，用于测试用例参数化
    indexes_can_append = [
        pd.RangeIndex(3),  # 创建一个 RangeIndex，范围为 0 到 2
        Index([4, 5, 6]),  # 创建一个普通的 Index，包含整数 4、5、6
        Index([4.5, 5.5, 6.5]),  # 创建一个普通的 Index，包含浮点数 4.5、5.5、6.5
        Index(list("abc")),  # 创建一个普通的 Index，包含字符 'a'、'b'、'c'
        pd.CategoricalIndex("A B C".split()),  # 创建一个分类索引，包含分类 'A'、'B'、'C'
        pd.CategoricalIndex("D E F".split(), ordered=True),  # 创建一个有序的分类索引，包含分类 'D'、'E'、'F'
        pd.IntervalIndex.from_breaks([7, 8, 9, 10]),  # 创建一个间隔索引，从断点 [7, 8, 9, 10] 创建
        pd.DatetimeIndex(  # 创建一个日期时间索引，包含三个日期时间对象
            [
                dt.datetime(2013, 1, 3, 0, 0),
                dt.datetime(2013, 1, 3, 6, 10),
                dt.datetime(2013, 1, 3, 7, 12),
            ]
        ),
        pd.MultiIndex.from_arrays(["A B C".split(), "D E F".split()]),  # 创建一个多级索引，包含两个级别
    ]

    @pytest.mark.parametrize(
        "index", indexes_can_append, ids=lambda x: type(x).__name__
    )
    def test_append_same_columns_type(self, index):
        # GH18359
        # 测试向 DataFrame 追加与原始数据列类型相同的 Series

        # 创建一个 DataFrame，比 Series 宽
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=index)
        ser_index = index[:2]  # 选取 Series 所需的一部分索引
        ser = Series([7, 8], index=ser_index, name=2)
        result = df._append(ser)  # 执行追加操作
        # 创建预期的 DataFrame
        expected = DataFrame(
            [[1, 2, 3.0], [4, 5, 6], [7, 8, np.nan]], index=[0, 1, 2], columns=index
        )
        # 检查整数 dtype 是否在 Series 的索引列中保留
        assert expected.dtypes.iloc[0].kind == "i"
        assert expected.dtypes.iloc[1].kind == "i"

        tm.assert_frame_equal(result, expected)  # 检查结果是否符合预期

        # 创建一个 Series，比 DataFrame 宽
        ser_index = index
        index = index[:2]  # 选取 DataFrame 所需的一部分索引
        df = DataFrame([[1, 2], [4, 5]], columns=index)
        ser = Series([7, 8, 9], index=ser_index, name=2)
        result = df._append(ser)  # 执行追加操作
        # 创建预期的 DataFrame
        expected = DataFrame(
            [[1, 2, np.nan], [4, 5, np.nan], [7, 8, 9]],
            index=[0, 1, 2],
            columns=ser_index,
        )
        tm.assert_frame_equal(result, expected)  # 检查结果是否符合预期

    @pytest.mark.parametrize(
        "df_columns, series_index",
        combinations(indexes_can_append, r=2),  # 对索引对象列表进行组合，用于测试不同类型的列
        ids=lambda x: type(x).__name__,
    )
    def test_append_different_columns_types(self, df_columns, series_index):
        # GH18359
        # 另见下面的 'test_append_different_columns_types_raises' 测试，用于追加时引发的错误

        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=df_columns)
        ser = Series([7, 8, 9], index=series_index, name=2)

        result = df._append(ser)  # 执行追加操作
        idx_diff = ser.index.difference(df_columns)  # 找到 Series 独有的索引
        combined_columns = Index(df_columns.tolist()).append(idx_diff)  # 组合列索引
        expected = DataFrame(
            [
                [1.0, 2.0, 3.0, np.nan, np.nan, np.nan],
                [4, 5, 6, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, 7, 8, 9],
            ],
            index=[0, 1, 2],
            columns=combined_columns,
        )
        tm.assert_frame_equal(result, expected)  # 检查结果是否符合预期
    # 定义测试方法，用于测试 DataFrame 的 _append 方法对数据类型转换的影响
    def test_append_dtype_coerce(self, sort):
        # GH 4993
        # 当添加带有 datetime 数据时，会错误地转换 datetime64 类型

        # 创建第一个 DataFrame，包含索引和 datetime 数据列
        df1 = DataFrame(
            index=[1, 2],
            data=[dt.datetime(2013, 1, 1, 0, 0), dt.datetime(2013, 1, 2, 0, 0)],
            columns=["start_time"],
        )
        
        # 创建第二个 DataFrame，包含索引和多个 datetime 数据列
        df2 = DataFrame(
            index=[4, 5],
            data=[
                [dt.datetime(2013, 1, 3, 0, 0), dt.datetime(2013, 1, 3, 6, 10)],
                [dt.datetime(2013, 1, 4, 0, 0), dt.datetime(2013, 1, 4, 7, 10)],
            ],
            columns=["start_time", "end_time"],
        )

        # 期望的合并结果，使用 concat 方法按列拼接为一个 DataFrame
        expected = concat(
            [
                Series(
                    [
                        pd.NaT,
                        pd.NaT,
                        dt.datetime(2013, 1, 3, 6, 10),
                        dt.datetime(2013, 1, 4, 7, 10),
                    ],
                    name="end_time",
                ),
                Series(
                    [
                        dt.datetime(2013, 1, 1, 0, 0),
                        dt.datetime(2013, 1, 2, 0, 0),
                        dt.datetime(2013, 1, 3, 0, 0),
                        dt.datetime(2013, 1, 4, 0, 0),
                    ],
                    name="start_time",
                ),
            ],
            axis=1,
            sort=sort,
        )

        # 使用 DataFrame 的 _append 方法将 df2 添加到 df1 中，并根据参数决定是否排序
        result = df1._append(df2, ignore_index=True, sort=sort)

        # 根据 sort 参数对期望结果进行列的排序
        if sort:
            expected = expected[["end_time", "start_time"]]
        else:
            expected = expected[["start_time", "end_time"]]

        # 使用 assert_frame_equal 断言结果 DataFrame 和期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试 DataFrame 的 _append 方法对缺失列正确的类型提升
    def test_append_missing_column_proper_upcast(self, sort):
        # 创建第一个 DataFrame，包含整型数组列 "A"
        df1 = DataFrame({"A": np.array([1, 2, 3, 4], dtype="i8")})
        
        # 创建第二个 DataFrame，包含布尔类型数组列 "B"
        df2 = DataFrame({"B": np.array([True, False, True, False], dtype=bool)})

        # 使用 DataFrame 的 _append 方法将 df2 添加到 df1 中，并忽略索引，根据参数决定是否排序
        appended = df1._append(df2, ignore_index=True, sort=sort)

        # 断言添加后的 DataFrame 列 "A" 的数据类型为 "f8"（浮点数类型）
        assert appended["A"].dtype == "f8"
        # 断言添加后的 DataFrame 列 "B" 的数据类型为 "O"（对象类型）
        assert appended["B"].dtype == "O"
    def test_append_empty_frame_to_series_with_dateutil_tz(self):
        # 定义一个测试方法：将空的 Series 追加到具有 dateutil 时区的 DataFrame 中
        # GH 23682

        # 创建一个带有特定时区的时间戳对象
        date = Timestamp("2018-10-24 07:30:00", tz=dateutil.tz.tzutc())

        # 创建一个 Series 对象，包含三个键值对
        ser = Series({"a": 1.0, "b": 2.0, "date": date})

        # 创建一个空的 DataFrame，列名为 "c", "d"
        df = DataFrame(columns=["c", "d"])

        # 在 DataFrame 上调用 _append 方法，将 ser 追加进去，并忽略索引
        result_a = df._append(ser, ignore_index=True)

        # 创建预期的 DataFrame 对象，包含特定的列和值
        expected = DataFrame(
            [[np.nan, np.nan, 1.0, 2.0, date]], columns=["c", "d", "a", "b", "date"]
        )

        # 将 "c" 和 "d" 列的类型转换为对象类型
        expected["c"] = expected["c"].astype(object)
        expected["d"] = expected["d"].astype(object)

        # 使用测试工具比较 result_a 和 expected 的内容是否相等
        tm.assert_frame_equal(result_a, expected)

        # 创建另一个预期的 DataFrame 对象，重复两次相同的行
        expected = DataFrame(
            [[np.nan, np.nan, 1.0, 2.0, date]] * 2, columns=["c", "d", "a", "b", "date"]
        )

        # 将 "c" 和 "d" 列的类型转换为对象类型
        expected["c"] = expected["c"].astype(object)
        expected["d"] = expected["d"].astype(object)

        # 在 result_a 上调用 _append 方法，将 ser 再次追加进去，并忽略索引
        result_b = result_a._append(ser, ignore_index=True)

        # 使用测试工具比较 result_b 和 expected 的内容是否相等
        tm.assert_frame_equal(result_b, expected)

        # 在 DataFrame 上调用 _append 方法，将包含两个 ser 的列表追加进去，并忽略索引
        result = df._append([ser, ser], ignore_index=True)

        # 使用测试工具比较 result 和 expected 的内容是否相等
        tm.assert_frame_equal(result, expected)

    def test_append_empty_tz_frame_with_datetime64ns(self):
        # 定义一个测试方法：将带有 datetime64ns 类型的空 DataFrame 追加
        # https://github.com/pandas-dev/pandas/issues/35460

        # 创建一个列名为 "a" 的空 DataFrame，并将其类型设置为 datetime64[ns, UTC]
        df = DataFrame(columns=["a"]).astype("datetime64[ns, UTC]")

        # 在 DataFrame 上调用 _append 方法，追加一个包含 pd.NaT 值的字典，并忽略索引
        result = df._append({"a": pd.NaT}, ignore_index=True)

        # 创建预期的 DataFrame 对象，包含一个列 "a"，其值为 pd.NaT
        expected = DataFrame({"a": [pd.NaT]}, dtype=object)

        # 使用测试工具比较 result 和 expected 的内容是否相等
        tm.assert_frame_equal(result, expected)

        # 创建另一个 Series 对象，包含一个键值对 {"a": pd.NaT}，并将其类型设置为 datetime64[ns]
        other = Series({"a": pd.NaT}, dtype="datetime64[ns]")

        # 在 DataFrame 上调用 _append 方法，追加这个 Series 对象，并忽略索引
        result = df._append(other, ignore_index=True)

        # 使用测试工具比较 result 和 expected 的内容是否相等
        tm.assert_frame_equal(result, expected)

        # 创建一个 Series 对象，其值为 pd.NaT，但类型设置为 datetime64[ns, US/Pacific]
        other = Series({"a": pd.NaT}, dtype="datetime64[ns, US/Pacific]")

        # 在 DataFrame 上调用 _append 方法，追加这个 Series 对象，并忽略索引
        result = df._append(other, ignore_index=True)

        # 创建预期的 DataFrame 对象，包含一个列 "a"，其值为 pd.NaT，并将类型设置为对象类型
        expected = DataFrame({"a": [pd.NaT]}).astype(object)

        # 使用测试工具比较 result 和 expected 的内容是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype_str", ["datetime64[ns, UTC]", "datetime64[ns]", "Int64", "int64"]
    )
    @pytest.mark.parametrize("val", [1, "NaT"])
    def test_append_empty_frame_with_timedelta64ns_nat(self, dtype_str, val):
        # 定义一个测试方法：将带有 timedelta64ns 或整数类型的空 DataFrame 追加
        # https://github.com/pandas-dev/pandas/issues/35460

        # 创建一个列名为 "a" 的空 DataFrame，并将其类型设置为 dtype_str 参数指定的类型
        df = DataFrame(columns=["a"]).astype(dtype_str)

        # 创建一个包含一个 timedelta64 值的 DataFrame 对象
        other = DataFrame({"a": [np.timedelta64(val, "ns")]})

        # 在 DataFrame 上调用 _append 方法，追加 other 对象，并忽略索引
        result = df._append(other, ignore_index=True)

        # 将预期的 DataFrame 对象类型转换为对象类型
        expected = other.astype(object)

        # 使用测试工具比较 result 和 expected 的内容是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype_str", ["datetime64[ns, UTC]", "datetime64[ns]", "Int64", "int64"]
    )
    @pytest.mark.parametrize("val", [1, "NaT"])
    # 定义测试方法，用于测试在具有 `timedelta64ns` 类型和 `val` 值的情况下追加数据帧
    def test_append_frame_with_timedelta64ns_nat(self, dtype_str, val):
        # 引用GitHub上的问题链接，说明该测试方法是为了解决特定问题而设计的
        # 创建一个包含单个元素的数据帧 `df`，元素类型为 `dtype_str`
        df = DataFrame({"a": pd.array([1], dtype=dtype_str)})

        # 创建另一个数据帧 `other`，包含一个 `np.timedelta64(val, "ns")` 的时间间隔对象
        other = DataFrame({"a": [np.timedelta64(val, "ns")]})

        # 调用 `_append` 方法将 `other` 数据帧追加到 `df` 中，忽略索引
        result = df._append(other, ignore_index=True)

        # 创建预期的结果数据帧 `expected`，包含 `df` 和 `other` 数据帧的第一个元素
        expected = DataFrame({"a": [df.iloc[0, 0], other.iloc[0, 0]]}, dtype=object)

        # 使用 `tm.assert_frame_equal` 断言函数来比较 `result` 和 `expected` 数据帧是否相等
        tm.assert_frame_equal(result, expected)
```