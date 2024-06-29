# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_combine_first.py`

```
from datetime import datetime  # 导入 datetime 模块中的 datetime 类

import numpy as np  # 导入 numpy 库并使用别名 np
import pytest  # 导入 pytest 测试框架

from pandas.core.dtypes.cast import find_common_type  # 从 pandas 库的 core.dtypes.cast 模块导入 find_common_type 函数
from pandas.core.dtypes.common import is_dtype_equal  # 从 pandas 库的 core.dtypes.common 模块导入 is_dtype_equal 函数

import pandas as pd  # 导入 pandas 库并使用别名 pd
from pandas import (  # 从 pandas 库中导入 DataFrame, Index, MultiIndex, Series 类
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm  # 导入 pandas 库中的 _testing 模块并使用别名 tm


class TestDataFrameCombineFirst:  # 定义 TestDataFrameCombineFirst 类
    def test_combine_first_mixed(self):  # 定义测试方法 test_combine_first_mixed，self 参数表示实例对象
        a = Series(["a", "b"], index=range(2))  # 创建 Series 对象 a，包含字符串 "a" 和 "b"
        b = Series(range(2), index=range(2))  # 创建 Series 对象 b，包含整数 0 和 1
        f = DataFrame({"A": a, "B": b})  # 创建 DataFrame 对象 f，包含两列 A 和 B

        a = Series(["a", "b"], index=range(5, 7))  # 创建 Series 对象 a，包含字符串 "a" 和 "b"，索引为 5 和 6
        b = Series(range(2), index=range(5, 7))  # 创建 Series 对象 b，包含整数 0 和 1，索引为 5 和 6
        g = DataFrame({"A": a, "B": b})  # 创建 DataFrame 对象 g，包含两列 A 和 B

        exp = DataFrame({"A": list("abab"), "B": [0, 1, 0, 1]}, index=[0, 1, 5, 6])
        # 创建期望的 DataFrame 对象 exp，包含两列 A 和 B，数据和索引与预期匹配

        combined = f.combine_first(g)  # 调用 DataFrame 对象 f 的 combine_first 方法，与 DataFrame 对象 g 合并
        tm.assert_frame_equal(combined, exp)  # 使用 pandas._testing 模块中的 assert_frame_equal 方法比较 combined 和 exp
    # 测试函数：测试 DataFrame.combine_first() 方法的不同情况

    def test_combine_first(self, float_frame, using_infer_string):
        # disjoint（无交集的情况）
        head, tail = float_frame[:5], float_frame[5:]

        # 将两部分数据合并，以头部数据为主，补充尾部数据的缺失部分
        combined = head.combine_first(tail)

        # 根据合并后的索引重新排序浮点数数据框架
        reordered_frame = float_frame.reindex(combined.index)

        # 断言合并后的数据框架与重新排序后的数据框架相等
        tm.assert_frame_equal(combined, reordered_frame)

        # 断言合并后的列索引与原浮点数数据框架的列索引相等
        tm.assert_index_equal(combined.columns, float_frame.columns)

        # 断言合并后的"A"列与重新排序后的"A"列相等
        tm.assert_series_equal(combined["A"], reordered_frame["A"])

        # same index（相同索引的情况）
        fcopy = float_frame.copy()
        fcopy["A"] = 1
        del fcopy["C"]

        fcopy2 = float_frame.copy()
        fcopy2["B"] = 0
        del fcopy2["D"]

        # 将两个数据框架进行合并，以fcopy为主，补充fcopy2的缺失部分
        combined = fcopy.combine_first(fcopy2)

        # 断言合并后"A"列的所有值都等于1
        assert (combined["A"] == 1).all()

        # 断言合并后的"B"列与fcopy的"B"列相等
        tm.assert_series_equal(combined["B"], fcopy["B"])

        # 断言合并后的"C"列与fcopy2的"C"列相等
        tm.assert_series_equal(combined["C"], fcopy2["C"])

        # 断言合并后的"D"列与fcopy的"D"列相等
        tm.assert_series_equal(combined["D"], fcopy["D"])

        # overlap（部分重叠的情况）
        head, tail = reordered_frame[:10].copy(), reordered_frame
        head["A"] = 1

        # 将头部数据和尾部数据进行合并，以头部数据为主，补充尾部数据的缺失部分
        combined = head.combine_first(tail)

        # 断言合并后的前10行"A"列的所有值都等于1
        assert (combined["A"][:10] == 1).all()

        # reverse overlap（逆向部分重叠的情况）
        tail.iloc[:10, tail.columns.get_loc("A")] = 0

        # 将尾部数据和头部数据进行合并，以尾部数据为主，补充头部数据的缺失部分
        combined = tail.combine_first(head)

        # 断言合并后的前10行"A"列的所有值都等于0
        assert (combined["A"][:10] == 0).all()

        # no overlap（无重叠的情况）
        f = float_frame[:10]
        g = float_frame[10:]

        # 将f和g进行合并，以f为主，补充g的缺失部分
        combined = f.combine_first(g)

        # 断言合并后的前10行"A"列的索引与f的索引相等，并且值也相等
        tm.assert_series_equal(combined["A"].reindex(f.index), f["A"])

        # 断言合并后的剩余行"A"列的索引与g的索引相等，并且值也相等
        tm.assert_series_equal(combined["A"].reindex(g.index), g["A"])

        # corner cases（边缘情况）
        # 根据是否使用"infer_string"来决定是否会有FutureWarning
        warning = FutureWarning if using_infer_string else None

        # 使用combine_first()方法合并float_frame和空DataFrame，断言会产生警告并且结果与float_frame相等
        with tm.assert_produces_warning(warning, match="empty entries"):
            comb = float_frame.combine_first(DataFrame())
        tm.assert_frame_equal(comb, float_frame)

        # 使用combine_first()方法合并空DataFrame和float_frame，断言结果与float_frame按索引排序后相等
        comb = DataFrame().combine_first(float_frame)
        tm.assert_frame_equal(comb, float_frame.sort_index())

        # 使用combine_first()方法合并float_frame和具有指定索引的空DataFrame，断言结果中包含"faz"索引
        comb = float_frame.combine_first(DataFrame(index=["faz", "boo"]))
        assert "faz" in comb.index

        # #2525
        # 创建DataFrame df和df2，使用combine_first()方法合并它们，断言结果中包含"b"列
        df = DataFrame({"a": [1]}, index=[datetime(2012, 1, 1)])
        df2 = DataFrame(columns=["b"])
        result = df.combine_first(df2)
        assert "b" in result
    def test_combine_first_same_as_in_update(self):
        # 定义一个测试函数，用于测试 DataFrame 的 combine_first 方法
        # 创建一个 DataFrame 对象 df，包含两行数据，每行有四列，数据类型为 float 和 bool
        df = DataFrame(
            [[1.0, 2.0, False, True], [4.0, 5.0, True, False]],
            columns=["A", "B", "bool1", "bool2"],
        )

        # 创建另一个 DataFrame 对象 other，包含一行数据，两列，数据类型为 int
        other = DataFrame([[45, 45]], index=[0], columns=["A", "B"])

        # 使用 df 的 combine_first 方法将 other 合并到 df 中，并将结果赋给 result
        result = df.combine_first(other)

        # 使用测试框架中的 assert_frame_equal 方法比较 result 和 df 是否相等
        tm.assert_frame_equal(result, df)

        # 修改 df 中第一行第一列的值为 NaN
        df.loc[0, "A"] = np.nan

        # 再次使用 combine_first 方法将 other 合并到修改后的 df 中，并将结果赋给 result
        result = df.combine_first(other)

        # 恢复 df 中第一行第一列的值为 45
        df.loc[0, "A"] = 45

        # 使用测试框架中的 assert_frame_equal 方法比较 result 和 df 是否相等
        tm.assert_frame_equal(result, df)

    def test_combine_first_doc_example(self):
        # 定义一个测试函数，展示 combine_first 方法的文档示例
        # 创建一个 DataFrame 对象 df1，包含两列数据，列名为 "A" 和 "B"，包含 NaN 值
        df1 = DataFrame(
            {"A": [1.0, np.nan, 3.0, 5.0, np.nan], "B": [np.nan, 2.0, 3.0, np.nan, 6.0]}
        )

        # 创建一个 DataFrame 对象 df2，包含两列数据，列名为 "A" 和 "B"，包含 NaN 值
        df2 = DataFrame(
            {
                "A": [5.0, 2.0, 4.0, np.nan, 3.0, 7.0],
                "B": [np.nan, np.nan, 3.0, 4.0, 6.0, 8.0],
            }
        )

        # 使用 df1 的 combine_first 方法将 df2 合并到 df1 中，并将结果赋给 result
        result = df1.combine_first(df2)

        # 创建一个预期的 DataFrame 对象 expected，包含两列数据 "A" 和 "B"，并赋予预期的值
        expected = DataFrame({"A": [1, 2, 3, 5, 3, 7.0], "B": [np.nan, 2, 3, 4, 6, 8]})

        # 使用测试框架中的 assert_frame_equal 方法比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_combine_first_return_obj_type_with_bools(self):
        # 定义一个测试函数，用于测试在包含布尔值的情况下 combine_first 方法的返回类型
        # 创建一个 DataFrame 对象 df1，包含三行数据，每行有三列数据，包含 NaN 和布尔值
        df1 = DataFrame(
            [[np.nan, 3.0, True], [-4.6, np.nan, True], [np.nan, 7.0, False]]
        )

        # 创建一个 DataFrame 对象 df2，包含两行数据，每行有三列数据，包含 NaN 和布尔值
        df2 = DataFrame([[-42.6, np.nan, True], [-5.0, 1.6, False]], index=[1, 2])

        # 创建一个预期的 Series 对象 expected，包含三个布尔值，并定义名称和数据类型
        expected = Series([True, True, False], name=2, dtype=bool)

        # 使用 df1 的 combine_first 方法合并 df2 到 df1 中，并取出第二列，将结果赋给 result_12
        result_12 = df1.combine_first(df2)[2]

        # 使用测试框架中的 assert_series_equal 方法比较 result_12 和 expected 是否相等
        tm.assert_series_equal(result_12, expected)

        # 使用 df2 的 combine_first 方法合并 df1 到 df2 中，并取出第二列，将结果赋给 result_21
        result_21 = df2.combine_first(df1)[2]

        # 使用测试框架中的 assert_series_equal 方法比较 result_21 和 expected 是否相等
        tm.assert_series_equal(result_21, expected)

    @pytest.mark.parametrize(
        "data1, data2, data_expected",
        (
            (
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
                [pd.NaT, pd.NaT, pd.NaT],
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
            ),
            (
                [pd.NaT, pd.NaT, pd.NaT],
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
            ),
            (
                [datetime(2000, 1, 2), pd.NaT, pd.NaT],
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
                [datetime(2000, 1, 2), datetime(2000, 1, 2), datetime(2000, 1, 3)],
            ),
            (
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
                [datetime(2000, 1, 2), pd.NaT, pd.NaT],
                [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
            ),
        ),
    )
    def test_combine_first_convert_datatime_correctly(
        self, data1, data2, data_expected
    ):
        # GH 3593
        # 创建两个数据帧，每个帧包含一个列 'a'，并分别用 data1 和 data2 的数据填充
        df1, df2 = DataFrame({"a": data1}), DataFrame({"a": data2})
        # 使用 combine_first 方法将 df2 的数据填充到 df1 中，生成一个新的数据帧 result
        result = df1.combine_first(df2)
        # 创建预期的数据帧 expected，其列 'a' 包含 data_expected 的数据
        expected = DataFrame({"a": data_expected})
        # 断言 result 和 expected 相等
        tm.assert_frame_equal(result, expected)

    def test_combine_first_align_nan(self):
        # GH 7509 (not fixed)
        # 创建数据帧 dfa，包含一个日期时间列 'a' 和整数列 'b'
        dfa = DataFrame([[pd.Timestamp("2011-01-01"), 2]], columns=["a", "b"])
        # 创建数据帧 dfb，包含一个整数列 'b'
        dfb = DataFrame([[4], [5]], columns=["b"])
        # 断言 dfa 的列 'a' 的数据类型为 datetime64[s]
        assert dfa["a"].dtype == "datetime64[s]"
        # 断言 dfa 的列 'b' 的数据类型为 int64

        assert dfa["b"].dtype == "int64"

        # 使用 combine_first 方法将 dfb 的数据填充到 dfa 中，生成结果 res
        res = dfa.combine_first(dfb)
        # 创建预期的数据帧 exp，包含列 'a' 中的第一个值为指定日期，第二个值为 NaT，列 'b' 包含 2 和 5
        exp = DataFrame(
            {"a": [pd.Timestamp("2011-01-01"), pd.NaT], "b": [2, 5]},
            columns=["a", "b"],
        )
        # 断言结果 res 和预期 exp 相等
        tm.assert_frame_equal(res, exp)
        # 断言 res 的列 'a' 的数据类型为 datetime64[s]
        assert res["a"].dtype == "datetime64[s]"
        # TODO: this must be int64
        # 断言 res 的列 'b' 的数据类型为 int64
        assert res["b"].dtype == "int64"

        # 对 dfa 的前零行使用 combine_first 方法填充 dfb 的数据，生成结果 res
        res = dfa.iloc[:0].combine_first(dfb)
        # 创建预期的数据帧 exp，其列 'a' 和 'b' 均为 NaN
        exp = DataFrame({"a": [np.nan, np.nan], "b": [4, 5]}, columns=["a", "b"])
        # 断言结果 res 和预期 exp 相等
        tm.assert_frame_equal(res, exp)
        # TODO: this must be datetime64
        # 断言 res 的列 'a' 的数据类型为 float64
        assert res["a"].dtype == "float64"
        # TODO: this must be int64
        # 断言 res 的列 'b' 的数据类型为 int64
        assert res["b"].dtype == "int64"

    def test_combine_first_timezone(self, unit):
        # see gh-7630
        # 创建一个日期时间数据 data1，并将其转换为指定的单位 unit，然后局部化为 UTC 时区
        data1 = pd.to_datetime("20100101 01:01").tz_localize("UTC").as_unit(unit)
        # 创建数据帧 df1，包含列 'UTCdatetime' 和 'abc'，数据为 data1，索引为指定日期范围
        df1 = DataFrame(
            columns=["UTCdatetime", "abc"],
            data=data1,
            index=pd.date_range("20140627", periods=1, unit=unit),
        )
        # 创建一个日期时间数据 data2，并将其转换为指定的单位 unit，然后局部化为 UTC 时区
        data2 = pd.to_datetime("20121212 12:12").tz_localize("UTC").as_unit(unit)
        # 创建数据帧 df2，包含列 'UTCdatetime' 和 'xyz'，数据为 data2，索引为指定日期范围
        df2 = DataFrame(
            columns=["UTCdatetime", "xyz"],
            data=data2,
            index=pd.date_range("20140628", periods=1, unit=unit),
        )
        # 使用 combine_first 方法将 df1 中缺失的部分用 df2 的数据填充，生成结果 res
        res = df2[["UTCdatetime"]].combine_first(df1)
        # 创建预期的数据帧 exp，包含列 'UTCdatetime' 和 'abc'，与 df1 的数据相匹配
        exp = DataFrame(
            {
                "UTCdatetime": [
                    pd.Timestamp("2010-01-01 01:01", tz="UTC"),
                    pd.Timestamp("2012-12-12 12:12", tz="UTC"),
                ],
                "abc": [pd.Timestamp("2010-01-01 01:01:00", tz="UTC"), pd.NaT],
            },
            columns=["UTCdatetime", "abc"],
            index=pd.date_range("20140627", periods=2, freq="D", unit=unit),
            dtype=f"datetime64[{unit}, UTC]",
        )
        # 断言结果 res 的列 'UTCdatetime' 的数据类型与指定的单位 unit 匹配
        assert res["UTCdatetime"].dtype == f"datetime64[{unit}, UTC]"
        # 断言结果 res 的列 'abc' 的数据类型与指定的单位 unit 匹配
        assert res["abc"].dtype == f"datetime64[{unit}, UTC]"

        # 断言结果 res 和预期 exp 相等
        tm.assert_frame_equal(res, exp)

    def test_combine_first_timezone2(self, unit):
        # see gh-10567
        # 创建一个包含指定时区和单位的日期范围 dts1，并生成数据帧 df1
        dts1 = pd.date_range("2015-01-01", "2015-01-05", tz="UTC", unit=unit)
        df1 = DataFrame({"DATE": dts1})
        # 创建一个包含指定时区和单位的日期范围 dts2，并生成数据帧 df2
        dts2 = pd.date_range("2015-01-03", "2015-01-05", tz="UTC", unit=unit)
        df2 = DataFrame({"DATE": dts2})

        # 使用 combine_first 方法将 df2 的数据填充到 df1 中，生成结果 res
        res = df1.combine_first(df2)
        # 断言结果 res 和 df1 相等
        tm.assert_frame_equal(res, df1)
        # 断言结果 res 的列 'DATE' 的数据类型与指定的单位 unit 匹配
        assert res["DATE"].dtype == f"datetime64[{unit}, UTC]"
    def test_combine_first_timezone3(self, unit):
        # 创建第一个时间索引，带有时区信息，并转换为指定单位的时间单位
        dts1 = pd.DatetimeIndex(
            ["2011-01-01", "NaT", "2011-01-03", "2011-01-04"], tz="US/Eastern"
        ).as_unit(unit)
        # 创建第一个数据框，以'DATE'列为键，使用上述时间索引和指定的索引
        df1 = DataFrame({"DATE": dts1}, index=[1, 3, 5, 7])
        # 创建第二个时间索引，带有时区信息，并转换为指定单位的时间单位
        dts2 = pd.DatetimeIndex(
            ["2012-01-01", "2012-01-02", "2012-01-03"], tz="US/Eastern"
        ).as_unit(unit)
        # 创建第二个数据框，以'DATE'列为键，使用上述时间索引和指定的索引
        df2 = DataFrame({"DATE": dts2}, index=[2, 4, 5])

        # 使用combine_first方法合并两个数据框，保留第一个数据框的值，缺失的部分用第二个数据框的值填充
        res = df1.combine_first(df2)
        # 创建预期的时间索引，带有时区信息，并转换为指定单位的时间单位
        exp_dts = pd.DatetimeIndex(
            [
                "2011-01-01",
                "2012-01-01",
                "NaT",
                "2012-01-02",
                "2011-01-03",
                "2011-01-04",
            ],
            tz="US/Eastern",
        ).as_unit(unit)
        # 创建预期的数据框，以'DATE'列为键，使用预期的时间索引和指定的索引
        exp = DataFrame({"DATE": exp_dts}, index=[1, 2, 3, 4, 5, 7])
        # 使用assert_frame_equal比较实际结果和预期结果是否相等
        tm.assert_frame_equal(res, exp)

    def test_combine_first_timezone4(self, unit):
        # 创建第一个时间范围，带有时区信息，使用指定单位的时间单位
        dts1 = pd.date_range("2015-01-01", "2015-01-05", tz="US/Eastern", unit=unit)
        # 创建第一个数据框，以'DATE'列为键，使用上述时间范围
        df1 = DataFrame({"DATE": dts1})
        # 创建第二个时间范围，不带时区信息，使用指定单位的时间单位
        dts2 = pd.date_range("2015-01-03", "2015-01-05", unit=unit)
        # 创建第二个数据框，以'DATE'列为键，使用上述时间范围
        df2 = DataFrame({"DATE": dts2})

        # 使用combine_first方法合并两个数据框，保留第一个数据框的值，缺失的部分用第二个数据框的值填充
        res = df1.combine_first(df2)
        # 使用assert_frame_equal比较实际结果和第一个数据框是否相等
        tm.assert_frame_equal(res, df1)
        # 断言结果数据框中'DATE'列的数据类型是否符合预期
        assert res["DATE"].dtype == f"datetime64[{unit}, US/Eastern]"

    def test_combine_first_timezone5(self, unit):
        # 创建第一个时间范围，带有时区信息，使用指定单位的时间单位
        dts1 = pd.date_range("2015-01-01", "2015-01-02", tz="US/Eastern", unit=unit)
        # 创建第一个数据框，以'DATE'列为键，使用上述时间范围
        df1 = DataFrame({"DATE": dts1})
        # 创建第二个时间范围，不带时区信息，使用指定单位的时间单位
        dts2 = pd.date_range("2015-01-01", "2015-01-03", unit=unit)
        # 创建第二个数据框，以'DATE'列为键，使用上述时间范围
        df2 = DataFrame({"DATE": dts2})

        # 使用combine_first方法合并两个数据框，保留第一个数据框的值，缺失的部分用第二个数据框的值填充
        res = df1.combine_first(df2)
        # 创建预期的时间戳列表，带有时区信息
        exp_dts = [
            pd.Timestamp("2015-01-01", tz="US/Eastern"),
            pd.Timestamp("2015-01-02", tz="US/Eastern"),
            pd.Timestamp("2015-01-03"),
        ]
        # 创建预期的数据框，以'DATE'列为键，使用预期的时间戳列表
        exp = DataFrame({"DATE": exp_dts})
        # 使用assert_frame_equal比较实际结果和预期结果是否相等
        tm.assert_frame_equal(res, exp)
        # 断言结果数据框中'DATE'列的数据类型是否符合预期
        assert res["DATE"].dtype == "object"

    def test_combine_first_timedelta(self):
        # 创建第一个时间跨度索引，使用指定单位的时间单位
        data1 = pd.TimedeltaIndex(["1 day", "NaT", "3 day", "4day"])
        # 创建第一个数据框，以'TD'列为键，使用上述时间跨度索引和指定的索引
        df1 = DataFrame({"TD": data1}, index=[1, 3, 5, 7])
        # 创建第二个时间跨度索引，使用指定单位的时间单位
        data2 = pd.TimedeltaIndex(["10 day", "11 day", "12 day"])
        # 创建第二个数据框，以'TD'列为键，使用上述时间跨度索引和指定的索引
        df2 = DataFrame({"TD": data2}, index=[2, 4, 5])

        # 使用combine_first方法合并两个数据框，保留第一个数据框的值，缺失的部分用第二个数据框的值填充
        res = df1.combine_first(df2)
        # 创建预期的时间跨度索引
        exp_dts = pd.TimedeltaIndex(
            ["1 day", "10 day", "NaT", "11 day", "3 day", "4 day"]
        )
        # 创建预期的数据框，以'TD'列为键，使用预期的时间跨度索引和指定的索引
        exp = DataFrame({"TD": exp_dts}, index=[1, 2, 3, 4, 5, 7])
        # 使用assert_frame_equal比较实际结果和预期结果是否相等
        tm.assert_frame_equal(res, exp)
        # 断言结果数据框中'TD'列的数据类型是否符合预期
        assert res["TD"].dtype == "timedelta64[ns]"
    def test_combine_first_period(self):
        # 创建第一个 PeriodIndex 对象，包含日期和 NaT（Not a Time）值
        data1 = pd.PeriodIndex(["2011-01", "NaT", "2011-03", "2011-04"], freq="M")
        # 创建第一个 DataFrame，使用 data1 作为数据，指定索引
        df1 = DataFrame({"P": data1}, index=[1, 3, 5, 7])
        # 创建第二个 PeriodIndex 对象，包含不同日期
        data2 = pd.PeriodIndex(["2012-01-01", "2012-02", "2012-03"], freq="M")
        # 创建第二个 DataFrame，使用 data2 作为数据，指定索引
        df2 = DataFrame({"P": data2}, index=[2, 4, 5])

        # 使用 combine_first 方法合并 df1 和 df2
        res = df1.combine_first(df2)
        # 创建预期的 PeriodIndex 对象，包含合并后的日期
        exp_dts = pd.PeriodIndex(
            ["2011-01", "2012-01", "NaT", "2012-02", "2011-03", "2011-04"], freq="M"
        )
        # 创建预期的 DataFrame，使用 exp_dts 作为数据，指定索引
        exp = DataFrame({"P": exp_dts}, index=[1, 2, 3, 4, 5, 7])
        # 使用 assert_frame_equal 断言 res 和 exp 相等
        tm.assert_frame_equal(res, exp)
        # 断言 res['P'] 的数据类型与 data1 的数据类型相同
        assert res["P"].dtype == data1.dtype

        # 不同的频率（freq）
        # 创建不同频率的 PeriodIndex 对象
        dts2 = pd.PeriodIndex(["2012-01-01", "2012-01-02", "2012-01-03"], freq="D")
        # 更新 df2 使用 dts2 作为数据
        df2 = DataFrame({"P": dts2}, index=[2, 4, 5])

        # 再次使用 combine_first 方法合并 df1 和 更新后的 df2
        res = df1.combine_first(df2)
        # 创建更新后的预期 PeriodIndex 对象列表
        exp_dts = [
            pd.Period("2011-01", freq="M"),
            pd.Period("2012-01-01", freq="D"),
            pd.NaT,
            pd.Period("2012-01-02", freq="D"),
            pd.Period("2011-03", freq="M"),
            pd.Period("2011-04", freq="M"),
        ]
        # 创建更新后的预期 DataFrame，使用 exp_dts 作为数据，指定索引
        exp = DataFrame({"P": exp_dts}, index=[1, 2, 3, 4, 5, 7])
        # 使用 assert_frame_equal 断言 res 和 exp 相等
        tm.assert_frame_equal(res, exp)
        # 断言 res['P'] 的数据类型为对象型
        assert res["P"].dtype == "object"

    def test_combine_first_int(self):
        # GH14687 - 整数序列不完全对齐的情况

        # 创建第一个整数 DataFrame
        df1 = DataFrame({"a": [0, 1, 3, 5]}, dtype="int64")
        # 创建第二个整数 DataFrame
        df2 = DataFrame({"a": [1, 4]}, dtype="int64")

        # 使用 combine_first 方法合并 df1 和 df2
        result_12 = df1.combine_first(df2)
        # 创建预期的整数 DataFrame
        expected_12 = DataFrame({"a": [0, 1, 3, 5]})
        # 使用 assert_frame_equal 断言 result_12 和 expected_12 相等
        tm.assert_frame_equal(result_12, expected_12)

        # 使用 combine_first 方法合并 df2 和 df1
        result_21 = df2.combine_first(df1)
        # 创建预期的整数 DataFrame
        expected_21 = DataFrame({"a": [1, 4, 3, 5]})
        # 使用 assert_frame_equal 断言 result_21 和 expected_21 相等
        tm.assert_frame_equal(result_21, expected_21)

    @pytest.mark.parametrize("val", [1, 1.0])
    def test_combine_first_with_asymmetric_other(self, val):
        # 查看 gh-20699
        # 创建第一个包含数值的 DataFrame
        df1 = DataFrame({"isNum": [val]})
        # 创建第二个包含布尔值的 DataFrame
        df2 = DataFrame({"isBool": [True]})

        # 使用 combine_first 方法合并 df1 和 df2
        res = df1.combine_first(df2)
        # 创建预期的 DataFrame，包含数值和布尔值列
        exp = DataFrame({"isBool": [True], "isNum": [val]})

        # 使用 assert_frame_equal 断言 res 和 exp 相等
        tm.assert_frame_equal(res, exp)

    def test_combine_first_string_dtype_only_na(self, nullable_string_dtype):
        # GH: 37519
        # 创建包含字符串和 NA 值的 DataFrame
        df = DataFrame(
            {"a": ["962", "85"], "b": [pd.NA] * 2}, dtype=nullable_string_dtype
        )
        # 创建另一个包含字符串和 NA 值的 DataFrame
        df2 = DataFrame({"a": ["85"], "b": [pd.NA]}, dtype=nullable_string_dtype)
        # 将 df 和 df2 设置为索引为 ["a", "b"] 的 DataFrame
        df.set_index(["a", "b"], inplace=True)
        df2.set_index(["a", "b"], inplace=True)
        # 使用 combine_first 方法合并 df 和 df2
        result = df.combine_first(df2)
        # 创建预期的 DataFrame，包含字符串和 NA 值
        expected = DataFrame(
            {"a": ["962", "85"], "b": [pd.NA] * 2}, dtype=nullable_string_dtype
        ).set_index(["a", "b"])
        # 使用 assert_frame_equal 断言 result 和 expected 相等
        tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "scalar1, scalar2",
    [
        # 参数化测试用例，测试不同类型的标量对于函数的影响
        (datetime(2020, 1, 1), datetime(2020, 1, 2)),
        (pd.Period("2020-01-01", "D"), pd.Period("2020-01-02", "D")),
        (pd.Timedelta("89 days"), pd.Timedelta("60 min")),
        (pd.Interval(left=0, right=1), pd.Interval(left=2, right=3, closed="left")),
    ],
)
def test_combine_first_timestamp_bug(scalar1, scalar2, nulls_fixture):
    # GH28481
    # 获取空值的固定值
    na_value = nulls_fixture

    # 创建 DataFrame，其中包含两列，每列包含一个空值的固定值
    frame = DataFrame([[na_value, na_value]], columns=["a", "b"])
    # 创建另一个 DataFrame，其中包含两列，每列包含一个标量
    other = DataFrame([[scalar1, scalar2]], columns=["b", "c"])

    # 查找两个 DataFrame 列的公共数据类型
    common_dtype = find_common_type([frame.dtypes["b"], other.dtypes["b"]])

    # 根据数据类型比较或列的数据类型是否相等，选择特定的标量作为值
    if (
        is_dtype_equal(common_dtype, "object")
        or frame.dtypes["b"] == other.dtypes["b"]
        or frame.dtypes["b"].kind == frame.dtypes["b"].kind == "M"
    ):
        val = scalar1
    else:
        val = na_value

    # 将 other DataFrame 的数据填充到 frame DataFrame，生成新的 DataFrame
    result = frame.combine_first(other)

    # 创建预期的 DataFrame，包含填充后的值
    expected = DataFrame([[na_value, val, scalar2]], columns=["a", "b", "c"])

    # 将预期的 DataFrame 的 'b' 列转换为公共数据类型
    expected["b"] = expected["b"].astype(common_dtype)

    # 使用测试框架检查结果和预期是否相等
    tm.assert_frame_equal(result, expected)


def test_combine_first_timestamp_bug_NaT():
    # GH28481
    # 创建包含 NaT 值的 DataFrame
    frame = DataFrame([[pd.NaT, pd.NaT]], columns=["a", "b"])
    # 创建包含日期时间标量的 DataFrame
    other = DataFrame(
        [[datetime(2020, 1, 1), datetime(2020, 1, 2)]], columns=["b", "c"]
    )

    # 将 other DataFrame 的数据填充到 frame DataFrame，生成新的 DataFrame
    result = frame.combine_first(other)

    # 创建预期的 DataFrame，包含填充后的值
    expected = DataFrame(
        [[pd.NaT, datetime(2020, 1, 1), datetime(2020, 1, 2)]], columns=["a", "b", "c"]
    )

    # 使用测试框架检查结果和预期是否相等
    tm.assert_frame_equal(result, expected)


def test_combine_first_with_nan_multiindex():
    # gh-36562
    # 创建包含 NaN 的 MultiIndex
    mi1 = MultiIndex.from_arrays(
        [["b", "b", "c", "a", "b", np.nan], [1, 2, 3, 4, 5, 6]], names=["a", "b"]
    )
    df = DataFrame({"c": [1, 1, 1, 1, 1, 1]}, index=mi1)
    # 创建包含 Series 的 DataFrame
    mi2 = MultiIndex.from_arrays(
        [["a", "b", "c", "a", "b", "d"], [1, 1, 1, 1, 1, 1]], names=["a", "b"]
    )
    s = Series([1, 2, 3, 4, 5, 6], index=mi2)
    # 将 Series 数据填充到 df DataFrame，生成新的 DataFrame
    res = df.combine_first(DataFrame({"d": s}))

    # 创建预期的 MultiIndex
    mi_expected = MultiIndex.from_arrays(
        [
            ["a", "a", "a", "b", "b", "b", "b", "c", "c", "d", np.nan],
            [1, 1, 4, 1, 1, 2, 5, 1, 3, 1, 6],
        ],
        names=["a", "b"],
    )
    # 创建预期的 DataFrame，包含填充后的值
    expected = DataFrame(
        {
            "c": [np.nan, np.nan, 1, 1, 1, 1, 1, np.nan, 1, np.nan, 1],
            "d": [1.0, 4.0, np.nan, 2.0, 5.0, np.nan, np.nan, 3.0, np.nan, 6.0, np.nan],
        },
        index=mi_expected,
    )
    # 使用测试框架检查结果和预期是否相等
    tm.assert_frame_equal(res, expected)


def test_combine_preserve_dtypes():
    # GH7509
    # 创建包含字符串列和整数列的 DataFrame
    a_column = Series(["a", "b"], index=range(2))
    b_column = Series(range(2), index=range(2))
    df1 = DataFrame({"A": a_column, "B": b_column})

    # 创建包含字符串列和整数列的 DataFrame
    c_column = Series(["a", "b"], index=range(5, 7))
    b_column = Series(range(-1, 1), index=range(5, 7))
    df2 = DataFrame({"B": b_column, "C": c_column})
    # 创建预期的 DataFrame，包含列"A", "B", "C"，以及对应的数据和索引
    expected = DataFrame(
        {
            "A": ["a", "b", np.nan, np.nan],  # 列"A"包含字符串和缺失值
            "B": [0, 1, -1, 0],  # 列"B"包含整数
            "C": [np.nan, np.nan, "a", "b"],  # 列"C"包含字符串和缺失值
        },
        index=[0, 1, 5, 6],  # 指定 DataFrame 的索引
    )
    # 使用 df2 的数据填充 df1，生成一个新的 DataFrame
    combined = df1.combine_first(df2)
    # 断言合并后的 DataFrame combined 是否与预期的 DataFrame expected 相等
    tm.assert_frame_equal(combined, expected)
# 测试函数：将 df1 和 df2 的数据进行合并，优先使用 df1 的数据填充空缺值，保留 MultiIndex 结构
def test_combine_first_duplicates_rows_for_nan_index_values():
    # 创建带有 MultiIndex 的 DataFrame df1，包含列 'x'，并设定 index 的值包括 NaN
    df1 = DataFrame(
        {"x": [9, 10, 11]},
        index=MultiIndex.from_arrays([[1, 2, 3], [np.nan, 5, 6]], names=["a", "b"]),
    )

    # 创建带有 MultiIndex 的 DataFrame df2，包含列 'y'，并设定 index 的值包括 NaN
    df2 = DataFrame(
        {"y": [12, 13, 14]},
        index=MultiIndex.from_arrays([[1, 2, 4], [np.nan, 5, 7]], names=["a", "b"]),
    )

    # 创建期望的 DataFrame，结构与 df1 和 df2 类似，合并后将 NaN 值填充为期望值
    expected = DataFrame(
        {
            "x": [9.0, 10.0, 11.0, np.nan],
            "y": [12.0, 13.0, np.nan, 14.0],
        },
        index=MultiIndex.from_arrays(
            [[1, 2, 3, 4], [np.nan, 5, 6, 7]], names=["a", "b"]
        ),
    )

    # 使用 combine_first 方法将 df2 的数据填充到 df1 中，生成合并后的结果
    combined = df1.combine_first(df2)

    # 使用测试框架验证合并结果与期望是否相符
    tm.assert_frame_equal(combined, expected)


# 测试函数：测试 combine_first 方法不将 int64 类型强制转换为 float64 类型的情况
def test_combine_first_int64_not_cast_to_float64():
    # 创建两个简单的 DataFrame df_1 和 df_2，共享列 'A' 和 'B'，df_2 多出列 'C'
    df_1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df_2 = DataFrame({"A": [1, 20, 30], "B": [40, 50, 60], "C": [12, 34, 65]})
    
    # 使用 combine_first 方法将 df_2 的数据填充到 df_1 中
    result = df_1.combine_first(df_2)
    
    # 创建期望的 DataFrame，包含 df_1 和 df_2 共同的所有列及数据
    expected = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [12, 34, 65]})
    
    # 使用测试框架验证合并结果与期望是否相符
    tm.assert_frame_equal(result, expected)


# 测试函数：测试 combine_first 方法在 MultiIndex 丢失数据类型时的行为
def test_midx_losing_dtype():
    # 创建带有 MultiIndex 的 DataFrame df1 和 df2，共享列 'a'，并在 index 中包含 NaN
    midx = MultiIndex.from_arrays([[0, 0], [np.nan, np.nan]])
    midx2 = MultiIndex.from_arrays([[1, 1], [np.nan, np.nan]])
    df1 = DataFrame({"a": [None, 4]}, index=midx)
    df2 = DataFrame({"a": [3, 3]}, index=midx2)
    
    # 使用 combine_first 方法将 df2 的数据填充到 df1 中
    result = df1.combine_first(df2)
    
    # 创建期望的 DataFrame，包含合并后的数据及 MultiIndex 结构
    expected_midx = MultiIndex.from_arrays(
        [[0, 0, 1, 1], [np.nan, np.nan, np.nan, np.nan]]
    )
    expected = DataFrame({"a": [np.nan, 4, 3, 3]}, index=expected_midx)
    
    # 使用测试框架验证合并结果与期望是否相符
    tm.assert_frame_equal(result, expected)


# 测试函数：测试 combine_first 方法在空列的情况下的行为
def test_combine_first_empty_columns():
    # 创建两个空 DataFrame，分别具有不同的列名 'a', 'b' 和 'a', 'c'
    left = DataFrame(columns=["a", "b"])
    right = DataFrame(columns=["a", "c"])
    
    # 使用 combine_first 方法将 right 的数据填充到 left 中
    result = left.combine_first(right)
    
    # 创建期望的 DataFrame，包含合并后的所有列 'a', 'b', 'c'
    expected = DataFrame(columns=["a", "b", "c"])
    
    # 使用测试框架验证合并结果与期望是否相符
    tm.assert_frame_equal(result, expected)
```