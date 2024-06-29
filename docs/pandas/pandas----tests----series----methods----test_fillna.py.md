# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_fillna.py`

```
from datetime import (
    datetime,  # 导入 datetime 类
    timedelta,  # 导入 timedelta 类
    timezone,  # 导入 timezone 类
)

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 库

from pandas import (
    Categorical,  # 导入 Categorical 类
    DataFrame,  # 导入 DataFrame 类
    DatetimeIndex,  # 导入 DatetimeIndex 类
    NaT,  # 导入 NaT 值
    Period,  # 导入 Period 类
    Series,  # 导入 Series 类
    Timedelta,  # 导入 Timedelta 类
    Timestamp,  # 导入 Timestamp 类
    date_range,  # 导入 date_range 函数
    isna,  # 导入 isna 函数
    timedelta_range,  # 导入 timedelta_range 函数
)
import pandas._testing as tm  # 导入 pandas 测试模块
from pandas.core.arrays import period_array  # 导入 period_array 函数


class TestSeriesFillNA:
    def test_fillna_nat(self):
        series = Series([0, 1, 2, NaT._value], dtype="M8[ns]")

        filled = series.ffill()  # 使用前向填充填充缺失值
        filled2 = series.fillna(value=series.values[2])  # 使用指定值填充缺失值

        expected = series.copy()
        expected.iloc[3] = expected.iloc[2]  # 将期望的第四个元素设置为第三个元素的值

        tm.assert_series_equal(filled, expected)  # 断言填充后的 series 与期望相等
        tm.assert_series_equal(filled2, expected)  # 断言填充后的 series 与期望相等

        df = DataFrame({"A": series})
        filled = df.ffill()  # 使用前向填充填充缺失值
        filled2 = df.fillna(value=series.values[2])  # 使用指定值填充缺失值
        expected = DataFrame({"A": expected})
        tm.assert_frame_equal(filled, expected)  # 断言填充后的 DataFrame 与期望相等
        tm.assert_frame_equal(filled2, expected)  # 断言填充后的 DataFrame 与期望相等

        series = Series([NaT._value, 0, 1, 2], dtype="M8[ns]")

        filled = series.bfill()  # 使用后向填充填充缺失值
        filled2 = series.fillna(value=series[1])  # 使用指定值填充缺失值

        expected = series.copy()
        expected[0] = expected[1]  # 将期望的第一个元素设置为第二个元素的值

        tm.assert_series_equal(filled, expected)  # 断言填充后的 series 与期望相等
        tm.assert_series_equal(filled2, expected)  # 断言填充后的 series 与期望相等

        df = DataFrame({"A": series})
        filled = df.bfill()  # 使用后向填充填充缺失值
        filled2 = df.fillna(value=series[1])  # 使用指定值填充缺失值
        expected = DataFrame({"A": expected})
        tm.assert_frame_equal(filled, expected)  # 断言填充后的 DataFrame 与期望相等
        tm.assert_frame_equal(filled2, expected)  # 断言填充后的 DataFrame 与期望相等

    def test_fillna(self):
        ts = Series(
            [0.0, 1.0, 2.0, 3.0, 4.0], index=date_range("2020-01-01", periods=5)
        )

        tm.assert_series_equal(ts, ts.ffill())  # 断言填充后的 series 与自身相等

        ts.iloc[2] = np.nan  # 将第三个元素设置为 NaN

        exp = Series([0.0, 1.0, 1.0, 3.0, 4.0], index=ts.index)
        tm.assert_series_equal(ts.ffill(), exp)  # 断言前向填充后的 series 与期望相等

        exp = Series([0.0, 1.0, 3.0, 3.0, 4.0], index=ts.index)
        tm.assert_series_equal(ts.bfill(), exp)  # 断言后向填充后的 series 与期望相等

        exp = Series([0.0, 1.0, 5.0, 3.0, 4.0], index=ts.index)
        tm.assert_series_equal(ts.fillna(value=5), exp)  # 断言指定值填充后的 series 与期望相等
    # 定义一个测试函数，用于测试 Series 对象的 fillna 方法在处理非标量填充值时的行为
    def test_fillna_nonscalar(self):
        # GH#5703: GitHub issue reference
        # 创建包含一个 NaN 值的 Series 对象 s1
        s1 = Series([np.nan])
        # 创建包含整数 1 的 Series 对象 s2
        s2 = Series([1])
        # 使用 s2 来填充 s1 中的 NaN 值，生成结果 Series 对象 result
        result = s1.fillna(s2)
        # 创建期望的结果 Series 对象 expected，其中 NaN 被填充为 1.0
        expected = Series([1.0])
        # 断言 result 与 expected 相等
        tm.assert_series_equal(result, expected)
        # 使用空字典来填充 s1，期望结果与 s1 保持一致
        result = s1.fillna({})
        tm.assert_series_equal(result, s1)
        # 使用空的 Series 对象来填充 s1，期望结果与 s1 保持一致
        result = s1.fillna(Series((), dtype=object))
        tm.assert_series_equal(result, s1)
        # 使用 s1 来填充 s2，期望结果与 s2 保持一致
        result = s2.fillna(s1)
        tm.assert_series_equal(result, s2)
        # 使用字典 {0: 1} 来填充 s1，期望结果与 expected 相等
        result = s1.fillna({0: 1})
        tm.assert_series_equal(result, expected)
        # 使用字典 {1: 1} 来填充 s1，期望结果与原始的 Series 对象 s1 相等
        result = s1.fillna({1: 1})
        tm.assert_series_equal(result, Series([np.nan]))
        # 使用字典 {0: 1, 1: 1} 来填充 s1，期望结果与 expected 相等
        result = s1.fillna({0: 1, 1: 1})
        tm.assert_series_equal(result, expected)
        # 使用 Series 对象 {0: 1, 1: 1} 来填充 s1，期望结果与 expected 相等
        result = s1.fillna(Series({0: 1, 1: 1}))
        tm.assert_series_equal(result, expected)
        # 使用带有自定义索引 [4, 5] 的 Series 对象来填充 s1，期望结果与 s1 相等
        result = s1.fillna(Series({0: 1, 1: 1}, index=[4, 5]))
        tm.assert_series_equal(result, s1)

    # 定义一个测试函数，用于测试 Series 对象的 fillna 方法在对齐操作中的行为
    def test_fillna_aligns(self):
        # 创建具有不同索引的两个 Series 对象 s1 和 s2
        s1 = Series([0, 1, 2], list("abc"))
        s2 = Series([0, np.nan, 2], list("bac"))
        # 使用 s1 来填充 s2 中的 NaN 值，生成结果 Series 对象 result
        result = s2.fillna(s1)
        # 创建期望的结果 Series 对象 expected，其中 NaN 被填充为 s1 中对应索引位置的值
        expected = Series([0, 0, 2.0], list("bac"))
        # 断言 result 与 expected 相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，用于测试 Series 对象的 fillna 方法在限制填充次数时的行为
    def test_fillna_limit(self):
        # 创建具有 NaN 值的 Series 对象 ser，指定 limit=1 进行填充
        ser = Series(np.nan, index=[0, 1, 2])
        result = ser.fillna(999, limit=1)
        # 创建期望的结果 Series 对象 expected，只填充了一个 NaN 值
        expected = Series([999, np.nan, np.nan], index=[0, 1, 2])
        tm.assert_series_equal(result, expected)

        # 指定 limit=2 进行填充
        result = ser.fillna(999, limit=2)
        # 创建期望的结果 Series 对象 expected，填充了两个 NaN 值
        expected = Series([999, 999, np.nan], index=[0, 1, 2])
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，用于测试 Series 对象的 fillna 方法在不转换字符串的情况下的行为
    def test_fillna_dont_cast_strings(self):
        # GH#9043: GitHub issue reference
        # 确保能正确填充表示整数/浮点数值的字符串，而不引发错误或转换类型
        vals = ["0", "1.5", "-0.3"]
        for val in vals:
            # 创建 dtype="float64" 的 Series 对象 ser
            ser = Series([0, 1, np.nan, np.nan, 4], dtype="float64")
            # 使用字符串 val 来填充 ser，期望结果为 dtype="object" 的 Series 对象 expected
            result = ser.fillna(val)
            expected = Series([0, 1, val, val, 4], dtype="object")
            # 断言 result 与 expected 相等
            tm.assert_series_equal(result, expected)
    def test_fillna_consistency(self):
        # GH#16402
        # 测试 fillna 将带时区的 Timestamp 填充到无时区的情况，应该结果是对象类型

        ser = Series([Timestamp("20130101"), NaT])

        result = ser.fillna(Timestamp("20130101", tz="US/Eastern"))
        expected = Series(
            [Timestamp("20130101"), Timestamp("2013-01-01", tz="US/Eastern")],
            dtype="object",
        )
        tm.assert_series_equal(result, expected)

        result = ser.where([True, False], Timestamp("20130101", tz="US/Eastern"))
        tm.assert_series_equal(result, expected)

        result = ser.where([True, False], Timestamp("20130101", tz="US/Eastern"))
        tm.assert_series_equal(result, expected)

        # 使用非日期时间值进行填充
        result = ser.fillna("foo")
        expected = Series([Timestamp("20130101"), "foo"])
        tm.assert_series_equal(result, expected)

        # 赋值操作
        ser2 = ser.copy()
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            ser2[1] = "foo"
        tm.assert_series_equal(ser2, expected)

    def test_datetime64_fillna(self):
        ser = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130102"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        ser[2] = np.nan

        # 前向填充
        result = ser.ffill()
        expected = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        tm.assert_series_equal(result, expected)

        # 后向填充
        result = ser.bfill()
        expected = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130103 9:01:01"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "scalar",
        [
            False,
            True,
        ],
    )
    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_datetime64_fillna_mismatched_reso_no_rounding(self, tz, scalar):
        # GH#56410
        # 测试当填充不匹配分辨率的日期时间时，不进行舍入处理

        dti = date_range("2016-01-01", periods=3, unit="s", tz=tz)
        item = Timestamp("2016-02-03 04:05:06.789", tz=tz)
        vec = date_range(item, periods=3, unit="ms")

        exp_dtype = "M8[ms]" if tz is None else "M8[ms, UTC]"
        expected = Series([item, dti[1], dti[2]], dtype=exp_dtype)

        ser = Series(dti)
        ser[0] = NaT
        ser2 = ser.copy()

        res = ser.fillna(item)
        res2 = ser2.fillna(Series(vec))

        if scalar:
            tm.assert_series_equal(res, expected)
        else:
            tm.assert_series_equal(res2, expected)
    # 定义一个测试方法，用于测试当填充不匹配的时间分辨率而不进行舍入时的情况
    def test_timedelta64_fillna_mismatched_reso_no_rounding(self, scalar):
        # GH#56410
        # 创建一个时间差索引，从1970-01-01到指定日期，单位为秒
        tdi = date_range("2016-01-01", periods=3, unit="s") - Timestamp("1970-01-01")
        # 创建一个时间戳对象，表示特定日期时间与1970-01-01的时间差
        item = Timestamp("2016-02-03 04:05:06.789") - Timestamp("1970-01-01")
        # 创建一个时间差序列，以指定的日期时间为起点，单位为毫秒
        vec = timedelta_range(item, periods=3, unit="ms")

        # 期望的结果是一个时间差序列，类型为'm8[ms]'
        expected = Series([item, tdi[1], tdi[2]], dtype="m8[ms]")

        # 创建一个时间戳序列
        ser = Series(tdi)
        # 将序列中的第一个元素设置为NaT（Not a Time）
        ser[0] = NaT
        # 复制序列
        ser2 = ser.copy()

        # 使用特定的值填充序列中的缺失值
        res = ser.fillna(item)
        # 使用另一个序列填充另一个序列中的缺失值
        res2 = ser2.fillna(Series(vec))

        # 如果scalar为True，比较填充后的序列res和期望的结果expected是否相等；否则，比较res2和expected是否相等
        if scalar:
            tm.assert_series_equal(res, expected)
        else:
            tm.assert_series_equal(res2, expected)

    # 定义一个测试方法，用于测试在后向填充（backfill）时的日期时间处理
    def test_datetime64_fillna_backfill(self):
        # GH#6587
        # 确保在填充时将缺失值视为整数处理
        ser = Series([NaT, NaT, "2013-08-05 15:30:00.000001"], dtype="M8[ns]")

        # 期望的结果是一个时间戳序列，都填充为"2013-08-05 15:30:00.000001"，类型为'M8[ns]'
        expected = Series(
            [
                "2013-08-05 15:30:00.000001",
                "2013-08-05 15:30:00.000001",
                "2013-08-05 15:30:00.000001",
            ],
            dtype="M8[ns]",
        )
        # 执行后向填充操作
        result = ser.bfill()
        # 比较填充后的结果和期望的结果是否相等
        tm.assert_series_equal(result, expected)

    # 使用参数化测试装饰器，测试在带有时区的日期时间上执行填充操作时的行为
    @pytest.mark.parametrize("tz", ["US/Eastern", "Asia/Tokyo"])
    def test_fillna_dt64tz_with_method(self):
        # 带有时区的情况
        # GH#15855
        # 创建一个带有时区信息的时间戳序列
        ser = Series([Timestamp("2012-11-11 00:00:00+01:00"), NaT])
        # 期望的结果是填充后的时间戳序列，都为带有时区信息的"2012-11-11 00:00:00+01:00"
        exp = Series(
            [
                Timestamp("2012-11-11 00:00:00+01:00"),
                Timestamp("2012-11-11 00:00:00+01:00"),
            ]
        )
        # 执行前向填充操作并比较结果
        tm.assert_series_equal(ser.ffill(), exp)

        # 创建另一个带有时区信息的时间戳序列
        ser = Series([NaT, Timestamp("2012-11-11 00:00:00+01:00")])
        # 期望的结果是填充后的时间戳序列，都为带有时区信息的"2012-11-11 00:00:00+01:00"
        exp = Series(
            [
                Timestamp("2012-11-11 00:00:00+01:00"),
                Timestamp("2012-11-11 00:00:00+01:00"),
            ]
        )
        # 执行后向填充操作并比较结果
        tm.assert_series_equal(ser.bfill(), exp)

    # 定义一个测试方法，用于测试在填充pytimedelta对象时的行为
    def test_fillna_pytimedelta(self):
        # GH#8209
        # 创建一个包含NaN和pytimedelta的序列
        ser = Series([np.nan, Timedelta("1 days")], index=["A", "B"])

        # 使用timedelta(1)填充序列中的缺失值
        result = ser.fillna(timedelta(1))
        # 期望的结果是填充后的序列，值为Timedelta("1 days")，索引与原序列相同
        expected = Series(Timedelta("1 days"), index=["A", "B"])
        # 比较填充后的结果和期望的结果是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，用于测试在填充Period对象时的行为
    def test_fillna_period(self):
        # GH#13737
        # 创建一个包含Period对象的序列，其中一个元素为缺失值
        ser = Series([Period("2011-01", freq="M"), Period("NaT", freq="M")])

        # 使用Period("2012-01", freq="M")填充缺失值
        res = ser.fillna(Period("2012-01", freq="M"))
        # 期望的结果是填充后的序列，缺失值被填充为Period("2012-01", freq="M")
        exp = Series([Period("2011-01", freq="M"), Period("2012-01", freq="M")])
        # 比较填充后的结果和期望的结果是否相等
        tm.assert_series_equal(res, exp)
        # 确保填充后的序列数据类型为"Period[M]"
        assert res.dtype == "Period[M]"
    def test_fillna_dt64_timestamp(self, frame_or_series):
        ser = Series(
            [
                Timestamp("20130101"),  # 创建一个时间戳对象，表示2013年1月1日
                Timestamp("20130101"),  # 创建另一个时间戳对象，表示2013年1月1日
                Timestamp("20130102"),  # 创建一个时间戳对象，表示2013年1月2日
                Timestamp("20130103 9:01:01"),  # 创建一个时间戳对象，表示2013年1月3日9点1分1秒
            ]
        )
        ser[2] = np.nan  # 将序列中索引为2的位置设置为NaN

        obj = frame_or_series(ser)  # 使用frame_or_series函数创建一个对象，传入序列ser

        # reg fillna
        result = obj.fillna(Timestamp("20130104"))  # 使用时间戳对象"20130104"填充NaN值
        expected = Series(
            [
                Timestamp("20130101"),  # 预期的第一个时间戳对象，不变
                Timestamp("20130101"),  # 预期的第二个时间戳对象，不变
                Timestamp("20130104"),  # 预期的第三个时间戳对象，被填充为"20130104"
                Timestamp("20130103 9:01:01"),  # 预期的第四个时间戳对象，不变
            ]
        )
        expected = frame_or_series(expected)  # 使用frame_or_series函数创建预期结果对象
        tm.assert_equal(result, expected)  # 断言结果与预期相等

        result = obj.fillna(NaT)  # 使用NaT（Not a Time）填充NaN值
        expected = obj  # 预期结果与原始对象相同
        tm.assert_equal(result, expected)  # 断言结果与预期相等

    def test_fillna_dt64_non_nao(self):
        # GH#27419
        ser = Series([Timestamp("2010-01-01"), NaT, Timestamp("2000-01-01")])  # 创建包含时间戳和NaT的序列
        val = np.datetime64("1975-04-05", "ms")  # 创建一个np.datetime64对象，表示1975-04-05毫秒级别精度

        result = ser.fillna(val)  # 使用val填充序列中的NaN值
        expected = Series(
            [Timestamp("2010-01-01"),  # 预期的第一个时间戳对象，不变
             Timestamp("1975-04-05"),  # 预期的第二个时间戳对象，被填充为val对应的时间
             Timestamp("2000-01-01")]  # 预期的第三个时间戳对象，不变
        )
        tm.assert_series_equal(result, expected)  # 断言结果序列与预期序列相等

    def test_fillna_numeric_inplace(self):
        x = Series([np.nan, 1.0, np.nan, 3.0, np.nan], ["z", "a", "b", "c", "d"])  # 创建包含NaN值的数值序列
        y = x.copy()  # 复制序列x到y

        return_value = y.fillna(value=0, inplace=True)  # 使用0填充y中的NaN值，并且原地修改y
        assert return_value is None  # 确保返回值为None，表示原地修改成功

        expected = x.fillna(value=0)  # 使用0填充x中的NaN值，创建预期结果
        tm.assert_series_equal(y, expected)  # 断言y与预期结果相等

    # ---------------------------------------------------------------
    # CategoricalDtype

    @pytest.mark.parametrize(
        "fill_value, expected_output",
        [
            ("a", ["a", "a", "b", "a", "a"]),  # 填充值为字符串"a"，预期结果为对应填充后的序列
            ({1: "a", 3: "b", 4: "b"}, ["a", "a", "b", "b", "b"]),  # 使用字典进行填充，预期结果为对应填充后的序列
            ({1: "a"}, ["a", "a", "b", np.nan, np.nan]),  # 使用部分键值对填充，预期结果为对应填充后的序列
            ({1: "a", 3: "b"}, ["a", "a", "b", "b", np.nan]),  # 使用部分键值对填充，预期结果为对应填充后的序列
            (Series("a"), ["a", np.nan, "b", np.nan, np.nan]),  # 使用Series对象填充，预期结果为对应填充后的序列
            (Series("a", index=[1]), ["a", "a", "b", np.nan, np.nan]),  # 使用带索引的Series对象填充，预期结果为对应填充后的序列
            (Series({1: "a", 3: "b"}), ["a", "a", "b", "b", np.nan]),  # 使用键值对填充，预期结果为对应填充后的序列
            (Series(["a", "b"], index=[3, 4]), ["a", np.nan, "b", "a", "b"]),  # 使用带索引的Series对象填充，预期结果为对应填充后的序列
        ],
    )
    def test_fillna_categorical(self, fill_value, expected_output):
        # GH#17033
        # Test fillna for a Categorical series
        data = ["a", np.nan, "b", np.nan, np.nan]  # 创建包含NaN值的数据列表
        ser = Series(Categorical(data, categories=["a", "b"]))  # 使用Categorical类型创建序列，指定类别为["a", "b"]
        exp = Series(Categorical(expected_output, categories=["a", "b"]))  # 使用预期输出创建Categorical类型的序列
        result = ser.fillna(fill_value)  # 使用填充值填充序列中的NaN值
        tm.assert_series_equal(result, exp)  # 断言结果序列与预期序列相等
    # 使用 pytest 的 @parametrize 装饰器，为 test_fillna_categorical_with_new_categories 方法添加多组参数化测试用例
    @pytest.mark.parametrize(
        "fill_value, expected_output",
        [
            (["a", "b", "c", "d", "e"], ["a", "b", "b", "d", "e"]),  # 参数化测试用例：填充值和期望输出
            (["b", "d", "a", "d", "a"], ["a", "d", "b", "d", "a"]),  # 参数化测试用例：填充值和期望输出
            (
                Categorical(
                    ["b", "d", "a", "d", "a"], categories=["b", "c", "d", "e", "a"]
                ),
                ["a", "d", "b", "d", "a"],  # 参数化测试用例：类别型填充值和期望输出
            ),
        ],
    )
    # 测试方法：使用参数化测试验证填充缺失值后的行为
    def test_fillna_categorical_with_new_categories(self, fill_value, expected_output):
        # GH#26215
        # 测试用例编号，关联 GitHub 上的 issue 编号
        data = ["a", np.nan, "b", np.nan, np.nan]
        # 创建一个 Series 对象，包含类别数据
        ser = Series(Categorical(data, categories=["a", "b", "c", "d", "e"]))
        # 期望的输出 Series 对象，也包含类别数据
        exp = Series(Categorical(expected_output, categories=["a", "b", "c", "d", "e"]))
        # 创建填充值的 Series 对象
        fill_value = Series(fill_value)
        # 执行填充缺失值操作
        result = ser.fillna(fill_value)
        # 断言填充后的结果与期望输出相等
        tm.assert_series_equal(result, exp)

    # 测试方法：验证在特定情况下会引发异常
    def test_fillna_categorical_raises(self):
        data = ["a", np.nan, "b", np.nan, np.nan]
        # 创建一个包含类别数据的 Series 对象
        ser = Series(Categorical(data, categories=["a", "b"]))
        cat = ser._values

        msg = "Cannot setitem on a Categorical with a new category"
        # 验证当尝试在 Categorical 对象中填充新的类别时会引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            ser.fillna("d")

        msg2 = "Length of 'value' does not match."
        # 验证当填充值的长度与类别数据的长度不匹配时会引发 ValueError 异常
        with pytest.raises(ValueError, match=msg2):
            cat.fillna(Series("d"))

        with pytest.raises(TypeError, match=msg):
            # 验证当尝试用字典形式填充时会引发 TypeError 异常
            ser.fillna({1: "d", 3: "a"})

        msg = '"value" parameter must be a scalar or dict, but you passed a "list"'
        # 验证当填充值参数不是标量或字典时会引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            ser.fillna(["a", "b"])

        msg = '"value" parameter must be a scalar or dict, but you passed a "tuple"'
        # 验证当填充值参数不是标量或字典时会引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            ser.fillna(("a", "b"))

        msg = (
            '"value" parameter must be a scalar, dict '
            'or Series, but you passed a "DataFrame"'
        )
        # 验证当填充值参数不是标量、字典或 Series 时会引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            ser.fillna(DataFrame({1: ["a"], 3: ["b"]}))

    # 使用 pytest 的 @parametrize 装饰器，为 dtype 和 scalar 参数化测试
    @pytest.mark.parametrize("dtype", [float, "float32", "float64"])
    @pytest.mark.parametrize("scalar", [True, False])
    # 定义一个测试方法，用于测试 fillna 方法在浮点数强制转换时的行为
    def test_fillna_float_casting(self, dtype, any_real_numpy_dtype, scalar):
        # 创建一个包含 NaN 和浮点数的 Series 对象，指定数据类型为 dtype
        ser = Series([np.nan, 1.2], dtype=dtype)
        # 创建一个填充值的 Series 对象，数据类型为 any_real_numpy_dtype
        fill_values = Series([2, 2], dtype=any_real_numpy_dtype)
        # 如果 scalar 为 True，则将 fill_values 转换为与 dtype 相同的类型的标量值 2
        if scalar:
            fill_values = fill_values.dtype.type(2)

        # 使用 fillna 方法填充 NaN 值
        result = ser.fillna(fill_values)
        # 期望的结果 Series，填充后应该为 [2.0, 1.2]
        expected = Series([2.0, 1.2], dtype=dtype)
        # 断言填充后的结果与期望的结果相等
        tm.assert_series_equal(result, expected)

        # 对原始 Series 使用布尔掩码填充 NaN 值
        ser = Series([np.nan, 1.2], dtype=dtype)
        mask = ser.isna().to_numpy()
        ser[mask] = fill_values
        # 断言填充后的 Series 与期望的结果相等
        tm.assert_series_equal(ser, expected)

        # 使用 mask 方法填充 NaN 值
        ser = Series([np.nan, 1.2], dtype=dtype)
        ser.mask(mask, fill_values, inplace=True)
        # 断言填充后的 Series 与期望的结果相等
        tm.assert_series_equal(ser, expected)

        # 使用 where 方法根据掩码条件填充 NaN 值
        ser = Series([np.nan, 1.2], dtype=dtype)
        res = ser.where(~mask, fill_values)
        # 断言填充后的 Series 与期望的结果相等
        tm.assert_series_equal(res, expected)

    # 定义一个测试方法，用于测试 fillna 方法在使用字典填充时的行为
    def test_fillna_f32_upcast_with_dict(self):
        # 创建一个包含 NaN 和浮点数的 Series 对象，数据类型为 np.float32
        ser = Series([np.nan, 1.2], dtype=np.float32)
        # 使用字典填充 NaN 值
        result = ser.fillna({0: 1})
        # 期望的结果 Series，填充后应该为 [1.0, 1.2]
        expected = Series([1.0, 1.2], dtype=np.float32)
        # 断言填充后的结果与期望的结果相等
        tm.assert_series_equal(result, expected)

    # ---------------------------------------------------------------
    # 无效用法

    # 定义一个测试方法，用于测试 fillna 方法在填充列表时的错误处理行为
    def test_fillna_listlike_invalid(self):
        # 创建一个包含随机整数的 Series 对象
        ser = Series(np.random.default_rng(2).integers(-100, 100, 50))
        # 定义错误消息
        msg = '"value" parameter must be a scalar or dict, but you passed a "list"'
        # 使用 pytest 断言检查填充列表时是否抛出预期的 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            ser.fillna([1, 2])

        # 定义错误消息
        msg = '"value" parameter must be a scalar or dict, but you passed a "tuple"'
        # 使用 pytest 断言检查填充元组时是否抛出预期的 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            ser.fillna((1, 2))

    # 定义一个测试方法，用于测试 fillna 方法在同时指定 method 和 limit 参数时的错误处理行为
    def test_fillna_method_and_limit_invalid(self):
        # 创建一个包含 None 的 Series 对象
        ser = Series([1, 2, 3, None])
        # 定义错误消息的正则表达式
        msg = "|".join(
            [
                r"Cannot specify both 'value' and 'method'\.",
                "Limit must be greater than 0",
                "Limit must be an integer",
            ]
        )
        # 遍历不同的 limit 值，使用 pytest 断言检查填充时是否抛出预期的 ValueError 异常
        for limit in [-1, 0, 1.0, 2.0]:
            with pytest.raises(ValueError, match=msg):
                ser.fillna(1, limit=limit)
    def test_fillna_datetime64_with_timezone_tzinfo(self):
        # 解决 GitHub 上的问题链接：https://github.com/pandas-dev/pandas/issues/38851
        # 处理不同的时区信息（tzinfo），即使表示为 UTC 也被视为相等
        ser = Series(date_range("2020", periods=3, tz="UTC"))
        expected = ser.copy()  # 复制原始序列以备后用
        ser[1] = NaT  # 将第二个位置的值设为 NaT (Not a Time)
        result = ser.fillna(datetime(2020, 1, 2, tzinfo=timezone.utc))  # 使用指定的 UTC 时区信息填充缺失值
        tm.assert_series_equal(result, expected)  # 断言填充后的结果与预期的相等

        # 在 2.0 之前，对于混合的时区信息，我们会将其强制转换为对象类型；在 2.0 中，会保留数据类型
        ts = Timestamp("2000-01-01", tz="US/Pacific")
        ser2 = Series(ser._values.tz_convert("dateutil/US/Pacific"))  # 将序列的时区信息转换为指定的时区
        assert ser2.dtype.kind == "M"  # 断言序列的数据类型为日期时间
        result = ser2.fillna(ts)  # 使用指定的时间戳填充缺失值
        expected = Series(
            [ser2[0], ts.tz_convert(ser2.dtype.tz), ser2[2]],  # 构建预期的序列，保持数据类型
            dtype=ser2.dtype,
        )
        tm.assert_series_equal(result, expected)  # 断言填充后的结果与预期的相等

    @pytest.mark.parametrize(
        "input, input_fillna, expected_data, expected_categories",
        [
            (["A", "B", None, "A"], "B", ["A", "B", "B", "A"], ["A", "B"]),
            (["A", "B", np.nan, "A"], "B", ["A", "B", "B", "A"], ["A", "B"]),
        ],
    )
    def test_fillna_categorical_accept_same_type(
        self, input, input_fillna, expected_data, expected_categories
    ):
        # 解决 GitHub 上的问题 GH32414
        cat = Categorical(input)  # 创建分类变量对象
        ser = Series(cat).fillna(input_fillna)  # 使用指定值填充分类变量中的缺失值
        filled = cat.fillna(ser)  # 使用另一个序列对象填充分类变量中的缺失值
        result = cat.fillna(filled)  # 再次填充分类变量中的缺失值
        expected = Categorical(expected_data, categories=expected_categories)  # 创建预期的分类变量对象
        tm.assert_categorical_equal(result, expected)  # 断言填充后的结果与预期的相等
class TestFillnaPad:
    # 测试fillna方法中的bug修复
    def test_fillna_bug(self):
        # 创建包含缺失值的Series对象
        ser = Series([np.nan, 1.0, np.nan, 3.0, np.nan], ["z", "a", "b", "c", "d"])
        # 使用前向填充方法填充缺失值
        filled = ser.ffill()
        # 期望得到的Series对象，缺失值被正确填充
        expected = Series([np.nan, 1.0, 1.0, 3.0, 3.0], ser.index)
        tm.assert_series_equal(filled, expected)

        # 使用后向填充方法填充缺失值
        filled = ser.bfill()
        # 期望得到的Series对象，缺失值被正确填充
        expected = Series([1.0, 1.0, 3.0, 3.0, np.nan], ser.index)
        tm.assert_series_equal(filled, expected)

    # 测试在混合数据类型且无缺失数据时的fillna方法
    def test_ffill_mixed_dtypes_without_missing_data(self):
        # GH#14956
        series = Series([datetime(2015, 1, 1, tzinfo=timezone.utc), 1])
        # 使用前向填充方法填充Series对象
        result = series.ffill()
        tm.assert_series_equal(series, result)

    # 测试fillna方法中的NaN填充
    def test_pad_nan(self):
        # 创建包含NaN的Series对象，指定数据类型为float
        x = Series(
            [np.nan, 1.0, np.nan, 3.0, np.nan], ["z", "a", "b", "c", "d"], dtype=float
        )

        # 在原地进行前向填充操作，并检查返回值
        return_value = x.ffill(inplace=True)
        assert return_value is None

        # 期望得到的Series对象，缺失值被正确填充
        expected = Series(
            [np.nan, 1.0, 1.0, 3.0, 3.0], ["z", "a", "b", "c", "d"], dtype=float
        )
        # 检查前向填充后的部分Series是否与期望一致
        tm.assert_series_equal(x[1:], expected[1:])
        # 检查第一个元素是否仍然是NaN
        assert np.isnan(x.iloc[0]), np.isnan(expected.iloc[0])

    # 测试Series对象中fillna方法的limit参数
    def test_series_fillna_limit(self):
        # 创建一个带有随机数据的Series对象
        index = np.arange(10)
        s = Series(np.random.default_rng(2).standard_normal(10), index=index)

        # 将Series对象的前两个元素重新索引，并使用前向填充方法填充，并设置填充的限制
        result = s[:2].reindex(index)
        result = result.ffill(limit=5)

        # 期望得到的Series对象，前两个元素被前向填充，后三个元素为NaN
        expected = s[:2].reindex(index).ffill()
        expected[-3:] = np.nan
        tm.assert_series_equal(result, expected)

        # 将Series对象的后两个元素重新索引，并使用后向填充方法填充，并设置填充的限制
        result = s[-2:].reindex(index)
        result = result.bfill(limit=5)

        # 期望得到的Series对象，后两个元素被后向填充，前三个元素为NaN
        expected = s[-2:].reindex(index).bfill()
        expected[:3] = np.nan
        tm.assert_series_equal(result, expected)

    # 测试填充整数类型数据的fillna方法
    def test_series_pad_backfill_limit(self):
        # 创建一个带有随机数据的Series对象
        index = np.arange(10)
        s = Series(np.random.default_rng(2).standard_normal(10), index=index)

        # 将Series对象的前两个元素重新索引，并使用前向填充方法填充，同时设置填充的限制
        result = s[:2].reindex(index, method="pad", limit=5)

        # 期望得到的Series对象，前两个元素被前向填充，后三个元素为NaN
        expected = s[:2].reindex(index).ffill()
        expected[-3:] = np.nan
        tm.assert_series_equal(result, expected)

        # 将Series对象的后两个元素重新索引，并使用后向填充方法填充，同时设置填充的限制
        result = s[-2:].reindex(index, method="backfill", limit=5)

        # 期望得到的Series对象，后两个元素被后向填充，前三个元素为NaN
        expected = s[-2:].reindex(index).bfill()
        expected[:3] = np.nan
        tm.assert_series_equal(result, expected)

    # 测试在整数类型数据上进行fillna操作
    def test_fillna_int(self):
        # 创建一个带有随机整数数据的Series对象
        ser = Series(np.random.default_rng(2).integers(-100, 100, 50))
        # 在原地进行前向填充操作，并检查返回值
        return_value = ser.ffill(inplace=True)
        assert return_value is None
        # 检查前向填充后的Series对象是否与未在原地填充的结果一致
        tm.assert_series_equal(ser.ffill(inplace=False), ser)
    # 定义测试方法：解决 datetime64tz 填充 NaN 时的问题
    def test_datetime64tz_fillna_round_issue(self):
        # GH#14872：GitHub 上对应的 issue 编号

        # 创建一个时间序列，包括 NaT（Not a Time）、具体日期时间和时区信息的日期时间对象
        data = Series(
            [NaT, NaT, datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=timezone.utc)]
        )

        # 对数据进行向后填充缺失值的操作
        filled = data.bfill()

        # 预期的填充后的时间序列，包括具体日期时间和时区信息
        expected = Series(
            [
                datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=timezone.utc),
                datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=timezone.utc),
                datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=timezone.utc),
            ]
        )

        # 使用测试模块中的 assert_series_equal 方法验证填充后的结果与预期是否相符
        tm.assert_series_equal(filled, expected)

    # 定义测试方法：解决 fillna 方法在 Pandas Array 上的应用问题
    def test_fillna_parr(self):
        # GH-24537：GitHub 上对应的 issue 编号

        # 创建一个日期时间索引，从 Timestamp 的最大值减去 10 纳秒，包含 5 个时间点，频率为纳秒
        dti = date_range(
            Timestamp.max - Timedelta(nanoseconds=10), periods=5, freq="ns"
        )

        # 将日期时间索引转换为周期对象的 Series
        ser = Series(dti.to_period("ns"))

        # 将索引位置为 2 的值设置为 NaT（Not a Time）
        ser[2] = NaT

        # 创建一个 Pandas 时间戳数组，包含指定的时间戳对象和频率为纳秒
        arr = period_array(
            [
                Timestamp("2262-04-11 23:47:16.854775797"),
                Timestamp("2262-04-11 23:47:16.854775798"),
                Timestamp("2262-04-11 23:47:16.854775798"),
                Timestamp("2262-04-11 23:47:16.854775800"),
                Timestamp("2262-04-11 23:47:16.854775801"),
            ],
            freq="ns",
        )

        # 用前向填充的方式填充缺失值
        filled = ser.ffill()

        # 预期的填充后的 Series，使用之前创建的时间戳数组
        expected = Series(arr)

        # 使用测试模块中的 assert_series_equal 方法验证填充后的结果与预期是否相符
        tm.assert_series_equal(filled, expected)
# 使用 pytest.mark.parametrize 装饰器，定义了一个参数化测试函数，用于测试 Series 对象的填充方法
@pytest.mark.parametrize(
    "data, expected_data, method, kwargs",
    (
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 3.0, 3.0, 3.0, 7.0, np.nan, np.nan],
            "ffill",  # 填充方法为 forward fill（向前填充）
            {"limit_area": "inside"},  # 填充区域限制为内部
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 3.0, np.nan, np.nan, 7.0, np.nan, np.nan],
            "ffill",  # 填充方法为 forward fill（向前填充）
            {"limit_area": "inside", "limit": 1},  # 填充区域限制为内部，最大填充限制为1
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, 7.0],
            "ffill",  # 填充方法为 forward fill（向前填充）
            {"limit_area": "outside"},  # 填充区域限制为外部
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, np.nan],
            "ffill",  # 填充方法为 forward fill（向前填充）
            {"limit_area": "outside", "limit": 1},  # 填充区域限制为外部，最大填充限制为1
        ),
        (
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "ffill",  # 填充方法为 forward fill（向前填充）
            {"limit_area": "outside", "limit": 1},  # 填充区域限制为外部，最大填充限制为1
        ),
        (
            range(5),
            range(5),
            "ffill",  # 填充方法为 forward fill（向前填充）
            {"limit_area": "outside", "limit": 1},  # 填充区域限制为外部，最大填充限制为1
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 7.0, 7.0, 7.0, 7.0, np.nan, np.nan],
            "bfill",  # 填充方法为 backward fill（向后填充）
            {"limit_area": "inside"},  # 填充区域限制为内部
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, 7.0, 7.0, np.nan, np.nan],
            "bfill",  # 填充方法为 backward fill（向后填充）
            {"limit_area": "inside", "limit": 1},  # 填充区域限制为内部，最大填充限制为1
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [3.0, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan],
            "bfill",  # 填充方法为 backward fill（向后填充）
            {"limit_area": "outside"},  # 填充区域限制为外部
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan],
            "bfill",  # 填充方法为 backward fill（向后填充）
            {"limit_area": "outside", "limit": 1},  # 填充区域限制为外部，最大填充限制为1
        ),
    ),
)
def test_ffill_bfill_limit_area(data, expected_data, method, kwargs):
    # GH#56492
    s = Series(data)  # 创建 Series 对象 s，使用给定的数据 data
    expected = Series(expected_data)  # 创建预期结果的 Series 对象 expected，使用 expected_data
    result = getattr(s, method)(**kwargs)  # 调用 s 对象的指定填充方法（method 参数），传入 kwargs 作为参数
    tm.assert_series_equal(result, expected)  # 使用 pandas 的 assert_series_equal 函数比较 result 和 expected
```