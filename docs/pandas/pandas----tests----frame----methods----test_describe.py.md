# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_describe.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，并简化命名为 pd
from pandas import (  # 从 Pandas 中导入多个子模块和函数
    Categorical,
    DataFrame,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

class TestDataFrameDescribe:
    def test_describe_bool_in_mixed_frame(self):
        df = DataFrame(  # 创建一个 Pandas DataFrame 对象
            {
                "string_data": ["a", "b", "c", "d", "e"],  # 包含字符串数据的列
                "bool_data": [True, True, False, False, False],  # 包含布尔数据的列
                "int_data": [10, 20, 30, 40, 50],  # 包含整数数据的列
            }
        )

        # Integer data are included in .describe() output,
        # Boolean and string data are not.
        result = df.describe()  # 对 DataFrame 进行描述性统计，并存储结果
        expected = DataFrame(  # 创建预期的 DataFrame 结果
            {"int_data": [5, 30, df.int_data.std(), 10, 20, 30, 40, 50]},  # 包含整数列的统计数据
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],  # 指定结果 DataFrame 的索引
        )
        tm.assert_frame_equal(result, expected)  # 使用 Pandas 内部函数比较结果和预期值

        # Top value is a boolean value that is False
        result = df.describe(include=["bool"])  # 对布尔数据进行描述性统计

        expected = DataFrame(  # 创建预期的布尔数据描述结果
            {"bool_data": [5, 2, False, 3]},  # 包含布尔数据列的统计信息
            index=["count", "unique", "top", "freq"],  # 指定结果 DataFrame 的索引
        )
        tm.assert_frame_equal(result, expected)  # 使用 Pandas 内部函数比较结果和预期值

    def test_describe_empty_object(self):
        # GH#27183
        df = DataFrame({"A": [None, None]}, dtype=object)  # 创建一个包含空对象的 DataFrame
        result = df.describe()  # 对 DataFrame 进行描述性统计
        expected = DataFrame(  # 创建预期的 DataFrame 结果
            {"A": [0, 0, np.nan, np.nan]},  # 包含空对象列的统计信息
            dtype=object,  # 指定结果 DataFrame 的数据类型
            index=["count", "unique", "top", "freq"],  # 指定结果 DataFrame 的索引
        )
        tm.assert_frame_equal(result, expected)  # 使用 Pandas 内部函数比较结果和预期值

        result = df.iloc[:0].describe()  # 对空的切片 DataFrame 进行描述性统计
        tm.assert_frame_equal(result, expected)  # 使用 Pandas 内部函数比较结果和预期值

    def test_describe_bool_frame(self):
        # GH#13891
        df = DataFrame(  # 创建一个包含布尔数据的 DataFrame
            {
                "bool_data_1": [False, False, True, True],  # 第一个布尔数据列
                "bool_data_2": [False, True, True, True],  # 第二个布尔数据列
            }
        )
        result = df.describe()  # 对 DataFrame 进行描述性统计
        expected = DataFrame(  # 创建预期的 DataFrame 结果
            {"bool_data_1": [4, 2, False, 2], "bool_data_2": [4, 2, True, 3]},  # 包含布尔数据列的统计信息
            index=["count", "unique", "top", "freq"],  # 指定结果 DataFrame 的索引
        )
        tm.assert_frame_equal(result, expected)  # 使用 Pandas 内部函数比较结果和预期值

        df = DataFrame(  # 创建一个新的 DataFrame
            {
                "bool_data": [False, False, True, True, False],  # 包含布尔数据的列
                "int_data": [0, 1, 2, 3, 4],  # 包含整数数据的列
            }
        )
        result = df.describe()  # 对 DataFrame 进行描述性统计
        expected = DataFrame(  # 创建预期的 DataFrame 结果
            {"int_data": [5, 2, df.int_data.std(), 0, 1, 2, 3, 4]},  # 包含整数数据列的统计信息
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],  # 指定结果 DataFrame 的索引
        )
        tm.assert_frame_equal(result, expected)  # 使用 Pandas 内部函数比较结果和预期值

        df = DataFrame(  # 创建一个新的 DataFrame
            {"bool_data": [False, False, True, True], "str_data": ["a", "b", "c", "a"]}  # 包含布尔和字符串数据的列
        )
        result = df.describe()  # 对 DataFrame 进行描述性统计
        expected = DataFrame(  # 创建预期的 DataFrame 结果
            {"bool_data": [4, 2, False, 2], "str_data": [4, 3, "a", 2]},  # 包含布尔和字符串数据列的统计信息
            index=["count", "unique", "top", "freq"],  # 指定结果 DataFrame 的索引
        )
        tm.assert_frame_equal(result, expected)  # 使用 Pandas 内部函数比较结果和预期值
    # 定义一个测试方法，用于测试描述分类数据
    def test_describe_categorical(self):
        # 创建一个包含随机整数值的数据框
        df = DataFrame({"value": np.random.default_rng(2).integers(0, 10000, 100)})
        # 创建标签列表，每个标签表示一个数值范围
        labels = [f"{i} - {i + 499}" for i in range(0, 10000, 500)]
        # 创建分类变量，将标签用作值和类别
        cat_labels = Categorical(labels, labels)

        # 按照'value'列的值对数据框进行升序排序
        df = df.sort_values(by=["value"], ascending=True)
        # 将'value'列的值分组到指定的区间，并使用cat_labels作为标签
        df["value_group"] = pd.cut(
            df.value, range(0, 10500, 500), right=False, labels=cat_labels
        )
        # 将结果存储到变量cat中
        cat = df

        # 对分类数据进行描述统计，结果应当只包含一个列
        result = cat.describe()
        assert len(result.columns) == 1

        # 对Series对象进行描述统计，验证结果是否符合预期
        cat = Categorical(
            ["a", "b", "b", "b"], categories=["a", "b", "c"], ordered=True
        )
        s = Series(cat)
        result = s.describe()
        expected = Series([4, 2, "b", 3], index=["count", "unique", "top", "freq"])
        tm.assert_series_equal(result, expected)

        # 创建一个包含分类数据的Series，并将其合并到数据框中
        cat = Series(Categorical(["a", "b", "c", "c"]))
        df3 = DataFrame({"cat": cat, "s": ["a", "b", "c", "c"]})
        # 对数据框进行描述统计，验证分类数据和字符串数据的统计结果是否一致
        result = df3.describe()
        tm.assert_numpy_array_equal(result["cat"].values, result["s"].values)

    # 定义一个测试方法，用于测试空分类列的描述统计
    def test_describe_empty_categorical_column(self):
        # 创建一个只包含空分类列的数据框
        df = DataFrame({"empty_col": Categorical([])})
        # 对数据框进行描述统计，验证空分类列的统计结果是否符合预期
        result = df.describe()
        expected = DataFrame(
            {"empty_col": [0, 0, np.nan, np.nan]},
            index=["count", "unique", "top", "freq"],
            dtype="object",
        )
        tm.assert_frame_equal(result, expected)
        # 验证统计结果中是否使用NaN表示缺失值，而不是None
        assert np.isnan(result.iloc[2, 0])
        assert np.isnan(result.iloc[3, 0])

    # 定义一个测试方法，用于测试包含分类列的数据框的描述统计
    def test_describe_categorical_columns(self):
        # 创建一个数据框，包含整数和对象类型的列，并使用分类索引
        columns = pd.CategoricalIndex(["int1", "int2", "obj"], ordered=True, name="XXX")
        df = DataFrame(
            {
                "int1": [10, 20, 30, 40, 50],
                "int2": [10, 20, 30, 40, 50],
                "obj": ["A", 0, None, "X", 1],
            },
            columns=columns,
        )
        # 对数据框进行描述统计，验证结果是否符合预期
        result = df.describe()

        # 准备预期的描述统计结果数据框和列索引
        exp_columns = pd.CategoricalIndex(
            ["int1", "int2"],
            categories=["int1", "int2", "obj"],
            ordered=True,
            name="XXX",
        )
        expected = DataFrame(
            {
                "int1": [5, 30, df.int1.std(), 10, 20, 30, 40, 50],
                "int2": [5, 30, df.int2.std(), 10, 20, 30, 40, 50],
            },
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            columns=exp_columns,
        )

        # 验证实际结果和预期结果是否一致
        tm.assert_frame_equal(result, expected)
        tm.assert_categorical_equal(result.columns.values, expected.columns.values)
    # 定义一个测试函数，用于描述处理日期时间的列
    def test_describe_datetime_columns(self):
        # 创建一个包含日期时间索引的对象，设置频率为每月起始日，时区为美国东部，名称为XXX
        columns = pd.DatetimeIndex(
            ["2011-01-01", "2011-02-01", "2011-03-01"],
            freq="MS",
            tz="US/Eastern",
            name="XXX",
        )
        # 创建一个DataFrame对象，包含三列数据，其中第三列包含混合类型
        df = DataFrame(
            {
                0: [10, 20, 30, 40, 50],
                1: [10, 20, 30, 40, 50],
                2: ["A", 0, None, "X", 1],
            }
        )
        # 设置DataFrame的列名为之前定义的日期时间索引对象
        df.columns = columns
        # 对DataFrame进行描述性统计，保存结果
        result = df.describe()

        # 创建一个预期的日期时间索引对象，包含两个日期
        exp_columns = pd.DatetimeIndex(
            ["2011-01-01", "2011-02-01"], freq="MS", tz="US/Eastern", name="XXX"
        )
        # 创建一个预期的DataFrame对象，包含描述统计的预期结果
        expected = DataFrame(
            {
                0: [5, 30, df.iloc[:, 0].std(), 10, 20, 30, 40, 50],
                1: [5, 30, df.iloc[:, 1].std(), 10, 20, 30, 40, 50],
            },
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )
        # 设置预期DataFrame的列名为预期的日期时间索引对象
        expected.columns = exp_columns
        # 使用断言比较实际结果DataFrame和预期DataFrame是否相等
        tm.assert_frame_equal(result, expected)
        # 使用断言检查结果DataFrame的列频率是否为每月起始日
        assert result.columns.freq == "MS"
        # 使用断言检查结果DataFrame的列时区是否与预期DataFrame的列时区相同
        assert result.columns.tz == expected.columns.tz
    def test_describe_timedelta_values(self):
        # GH#6145
        # 创建时间间隔序列 t1，包含 5 个时间间隔，频率为每天一次
        t1 = pd.timedelta_range("1 days", freq="D", periods=5)
        # 创建时间间隔序列 t2，包含 5 个时间间隔，频率为每小时一次
        t2 = pd.timedelta_range("1 hours", freq="h", periods=5)
        # 创建 DataFrame df，包含两列 t1 和 t2
        df = DataFrame({"t1": t1, "t2": t2})

        # 创建预期的 DataFrame expected，包含 t1 和 t2 列的统计描述
        expected = DataFrame(
            {
                "t1": [
                    5,
                    pd.Timedelta("3 days"),
                    df.iloc[:, 0].std(),
                    pd.Timedelta("1 days"),
                    pd.Timedelta("2 days"),
                    pd.Timedelta("3 days"),
                    pd.Timedelta("4 days"),
                    pd.Timedelta("5 days"),
                ],
                "t2": [
                    5,
                    pd.Timedelta("3 hours"),
                    df.iloc[:, 1].std(),
                    pd.Timedelta("1 hours"),
                    pd.Timedelta("2 hours"),
                    pd.Timedelta("3 hours"),
                    pd.Timedelta("4 hours"),
                    pd.Timedelta("5 hours"),
                ],
            },
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )

        # 对 DataFrame df 进行描述性统计，保存结果到 result
        result = df.describe()
        # 使用 tm.assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 创建预期的输出字符串 exp_repr，描述 result 的内容
        exp_repr = (
            "                              t1                         t2\n"
            "count                          5                          5\n"
            "mean             3 days 00:00:00            0 days 03:00:00\n"
            "std    1 days 13:56:50.394919273  0 days 01:34:52.099788303\n"
            "min              1 days 00:00:00            0 days 01:00:00\n"
            "25%              2 days 00:00:00            0 days 02:00:00\n"
            "50%              3 days 00:00:00            0 days 03:00:00\n"
            "75%              4 days 00:00:00            0 days 04:00:00\n"
            "max              5 days 00:00:00            0 days 05:00:00"
        )
        # 断言 result 的字符串表示应与 exp_repr 相等
        assert repr(result) == exp_repr

    def test_describe_tz_values(self, tz_naive_fixture):
        # GH#21332
        # 从测试夹具中获取时区 tz
        tz = tz_naive_fixture
        # 创建 Series s1，包含整数 0 到 4
        s1 = Series(range(5))
        # 创建起始和结束时间戳
        start = Timestamp(2018, 1, 1)
        end = Timestamp(2018, 1, 5)
        # 创建 Series s2，包含在时区 tz 下从 start 到 end 的日期范围
        s2 = Series(date_range(start, end, tz=tz))
        # 创建 DataFrame df，包含两列 s1 和 s2
        df = DataFrame({"s1": s1, "s2": s2})

        # 创建预期的 DataFrame expected，包含 s1 和 s2 列的统计描述
        expected = DataFrame(
            {
                "s1": [5, 2, 0, 1, 2, 3, 4, 1.581139],
                "s2": [
                    5,
                    Timestamp(2018, 1, 3).tz_localize(tz),
                    start.tz_localize(tz),
                    s2[1],
                    s2[2],
                    s2[3],
                    end.tz_localize(tz),
                    np.nan,
                ],
            },
            index=["count", "mean", "min", "25%", "50%", "75%", "max", "std"],
        )

        # 对 DataFrame df 进行包括所有列的描述性统计，保存结果到 result
        result = df.describe(include="all")
        # 使用 tm.assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
    def test_datetime_is_numeric_includes_datetime(self):
        # 创建一个 DataFrame，包含列"a"为日期范围，列"b"为列表[1, 2, 3]
        df = DataFrame({"a": date_range("2012", periods=3), "b": [1, 2, 3]})
        # 对 DataFrame 进行描述性统计
        result = df.describe()
        
        # 期望的 DataFrame，包含了对列"a"的描述统计和特定日期时间戳的描述
        expected = DataFrame(
            {
                "a": [
                    3,  # 总计行数
                    Timestamp("2012-01-02"),  # 平均日期时间戳
                    Timestamp("2012-01-01"),  # 最小日期时间戳
                    Timestamp("2012-01-01T12:00:00"),  # 25% 分位数的日期时间戳
                    Timestamp("2012-01-02"),  # 50% 分位数的日期时间戳
                    Timestamp("2012-01-02T12:00:00"),  # 75% 分位数的日期时间戳
                    Timestamp("2012-01-03"),  # 最大日期时间戳
                    np.nan,  # 标准差行
                ],
                "b": [3, 2, 1, 1.5, 2, 2.5, 3, 1],  # 列"b"的描述统计
            },
            index=["count", "mean", "min", "25%", "50%", "75%", "max", "std"],  # 索引为统计项目
        )
        # 使用测试工具比较实际结果和期望结果的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_describe_tz_values2(self):
        # 设定时区为 "CET"
        tz = "CET"
        # 创建 Series "s1"，包含整数范围[0, 4]
        s1 = Series(range(5))
        # 设定开始日期时间戳为2018年1月1日，结束日期时间戳为2018年1月5日
        start = Timestamp(2018, 1, 1)
        end = Timestamp(2018, 1, 5)
        # 创建 Series "s2"，包含在指定时区下的日期范围
        s2 = Series(date_range(start, end, tz=tz))
        # 创建 DataFrame，包含列"s1"和"s2"
        df = DataFrame({"s1": s1, "s2": s2})

        # 分别对 Series "s1" 和 "s2" 进行描述性统计
        s1_ = s1.describe()
        s2_ = s2.describe()
        
        # 创建预期的 DataFrame，合并 "s1" 和 "s2" 的描述统计结果，并根据指定索引重新排序
        idx = [
            "count",
            "mean",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "std",
        ]
        expected = pd.concat([s1_, s2_], axis=1, keys=["s1", "s2"]).reindex(idx)

        # 对整体 DataFrame 进行描述性统计，并使用测试工具比较实际结果和期望结果的 DataFrame 是否相等
        result = df.describe(include="all")
        tm.assert_frame_equal(result, expected)

    def test_describe_percentiles_integer_idx(self):
        # GH#26660
        # 创建 DataFrame，包含列"x"，其中元素为整数1
        df = DataFrame({"x": [1]})
        # 生成等间隔的百分位数数组
        pct = np.linspace(0, 1, 10 + 1)
        # 对 DataFrame 进行描述性统计，指定百分位数
        result = df.describe(percentiles=pct)

        # 创建期望的 DataFrame，包含列"x"的描述统计结果，以及自定义的百分位数行
        expected = DataFrame(
            {"x": [1.0, 1.0, np.nan, 1.0, *(1.0 for _ in pct), 1.0]},
            index=[
                "count",
                "mean",
                "std",
                "min",
                "0%",
                "10%",
                "20%",
                "30%",
                "40%",
                "50%",
                "60%",
                "70%",
                "80%",
                "90%",
                "100%",
                "max",
            ],
        )
        # 使用测试工具比较实际结果和期望结果的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_describe_does_not_raise_error_for_dictlike_elements(self):
        # GH#32409
        # 创建包含字典元素的 DataFrame
        df = DataFrame([{"test": {"a": "1"}}, {"test": {"a": "2"}}])
        # 创建期望的 DataFrame，包含对字典元素的描述统计
        expected = DataFrame(
            {"test": [2, 2, {"a": "1"}, 1]}, index=["count", "unique", "top", "freq"]
        )
        # 对 DataFrame 进行描述性统计，并使用测试工具比较实际结果和期望结果的 DataFrame 是否相等
        result = df.describe()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("exclude", ["x", "y", ["x", "y"], ["x", "z"]])
    def test_describe_when_include_all_exclude_not_allowed(self, exclude):
        """
        When include is 'all', then setting exclude != None is not allowed.
        """
        # 创建一个包含单列的 DataFrame，用于测试 describe 方法
        df = DataFrame({"x": [1], "y": [2], "z": [3]})
        # 当 include 参数为 'all' 且 exclude 参数不为 None 时，抛出 ValueError 异常
        msg = "exclude must be None when include is 'all'"
        with pytest.raises(ValueError, match=msg):
            # 调用 describe 方法，测试 include 和 exclude 参数的组合
            df.describe(include="all", exclude=exclude)

    def test_describe_with_duplicate_columns(self):
        # 创建一个包含重复列名的 DataFrame，用于测试 describe 方法
        df = DataFrame(
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            columns=["bar", "a", "a"],
            dtype="float64",
        )
        # 调用 describe 方法，生成结果 DataFrame
        result = df.describe()
        # 获取第一列的 describe 结果
        ser = df.iloc[:, 0].describe()
        # 构建预期的 DataFrame，使用 concat 将多个描述结果合并到一起
        expected = pd.concat([ser, ser, ser], keys=df.columns, axis=1)
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_ea_with_na(self, any_numeric_ea_dtype):
        """
        Test for handling NA values in DataFrame describe method.
        """
        # 创建一个包含 NA 值的 DataFrame，用于测试 describe 方法
        df = DataFrame({"a": [1, pd.NA, pd.NA], "b": pd.NA}, dtype=any_numeric_ea_dtype)
        # 调用 describe 方法，生成结果 DataFrame
        result = df.describe()
        # 构建预期的 DataFrame，包含处理 NA 值后的统计结果
        expected = DataFrame(
            {"a": [1.0, 1.0, pd.NA] + [1.0] * 5, "b": [0.0] + [pd.NA] * 7},
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            dtype="Float64",
        )
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_describe_exclude_pa_dtype(self):
        """
        Test describe method with specific ArrowDtype inclusion and exclusion.
        """
        # 导入 pyarrow 库，如果不存在则跳过测试
        pa = pytest.importorskip("pyarrow")
        # 创建一个包含指定 ArrowDtype 的 DataFrame，用于测试 describe 方法
        df = DataFrame(
            {
                "a": Series([1, 2, 3], dtype=pd.ArrowDtype(pa.int8())),
                "b": Series([1, 2, 3], dtype=pd.ArrowDtype(pa.int16())),
                "c": Series([1, 2, 3], dtype=pd.ArrowDtype(pa.int32())),
            }
        )
        # 调用 describe 方法，生成结果 DataFrame，指定 include 和 exclude 参数
        result = df.describe(
            include=pd.ArrowDtype(pa.int8()), exclude=pd.ArrowDtype(pa.int32())
        )
        # 构建预期的 DataFrame，包含特定 ArrowDtype 的统计结果
        expected = DataFrame(
            {"a": [3, 2, 1, 1, 1.5, 2, 2.5, 3]},
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            dtype=pd.ArrowDtype(pa.float64()),
        )
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
```