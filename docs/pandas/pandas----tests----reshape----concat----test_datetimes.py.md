# `D:\src\scipysrc\pandas\pandas\tests\reshape\concat\test_datetimes.py`

```
# 导入 datetime 模块并将其命名为 dt
import datetime as dt
# 从 datetime 模块中导入 datetime 类
from datetime import datetime

# 导入 dateutil 库
import dateutil
# 导入 numpy 库并将其命名为 np
import numpy as np
# 导入 pytest 库
import pytest

# 导入 pandas 库并将其命名为 pd
import pandas as pd
# 从 pandas 库中导入多个类和函数
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
    to_timedelta,
)
# 导入 pandas 内部的测试工具
import pandas._testing as tm


# 定义测试类 TestDatetimeConcat
class TestDatetimeConcat:
    # 定义测试方法 test_concat_datetime64_block
    def test_concat_datetime64_block(self):
        # 创建一个日期范围对象，从 "1/1/2000" 开始，包含 10 个时间点
        rng = date_range("1/1/2000", periods=10)

        # 创建一个 DataFrame 对象，列名为 'time'，数据为日期范围 rng
        df = DataFrame({"time": rng})

        # 对 df 进行连接操作，将两个 df 对象连接起来
        result = concat([df, df])
        # 断言：连接后的结果的前 10 行的 'time' 列应与 rng 相等
        assert (result.iloc[:10]["time"] == rng).all()
        # 断言：连接后的结果的第 11 到最后的 'time' 列应与 rng 相等
        assert (result.iloc[10:]["time"] == rng).all()

    # 定义测试方法 test_concat_datetime_datetime64_frame
    def test_concat_datetime_datetime64_frame(self):
        # GH#2624 说明此处是为了解决某个 GitHub 上的 issue 编号为 2624

        # 创建一个空列表 rows
        rows = []
        # 向 rows 列表中添加两个列表，分别包含日期时间对象和数据
        rows.append([datetime(2010, 1, 1), 1])
        rows.append([datetime(2010, 1, 2), "hi"])

        # 使用 from_records 方法从 rows 列表创建 DataFrame 对象 df2_obj
        df2_obj = DataFrame.from_records(rows, columns=["date", "test"])

        # 创建一个日期范围对象 ind，从 "2000/1/1" 开始，按天递增，共 10 个时间点
        ind = date_range(start="2000/1/1", freq="D", periods=10)
        # 创建一个 DataFrame 对象 df1，包含 'date' 和 'test' 列
        df1 = DataFrame({"date": ind, "test": range(10)})

        # 进行连接操作，将 df1 和 df2_obj 连接起来
        concat([df1, df2_obj])
    def test_concat_datetime_timezone(self):
        # GH 18523
        # 创建具有指定时区的日期范围索引 idx1
        idx1 = date_range("2011-01-01", periods=3, freq="h", tz="Europe/Paris")
        # 根据 idx1 的起始和结束创建日期范围索引 idx2
        idx2 = date_range(start=idx1[0], end=idx1[-1], freq="h")
        # 创建包含在 idx1 索引上的 DataFrame df1
        df1 = DataFrame({"a": [1, 2, 3]}, index=idx1)
        # 创建包含在 idx2 索引上的 DataFrame df2
        df2 = DataFrame({"b": [1, 2, 3]}, index=idx2)
        # 将 df1 和 df2 沿列方向连接起来
        result = concat([df1, df2], axis=1)

        # 期望的日期时间索引 exp_idx，带有指定时区
        exp_idx = DatetimeIndex(
            [
                "2011-01-01 00:00:00+01:00",
                "2011-01-01 01:00:00+01:00",
                "2011-01-01 02:00:00+01:00",
            ],
            dtype="M8[ns, Europe/Paris]",
            freq="h",
        )
        # 创建期望的 DataFrame，包含预期的数据和索引
        expected = DataFrame(
            [[1, 1], [2, 2], [3, 3]], index=exp_idx, columns=["a", "b"]
        )

        # 断言 result 和 expected 的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 创建具有不同时区的日期范围索引 idx3
        idx3 = date_range("2011-01-01", periods=3, freq="h", tz="Asia/Tokyo")
        # 创建包含在 idx3 索引上的 DataFrame df3
        df3 = DataFrame({"b": [1, 2, 3]}, index=idx3)
        # 将 df1 和 df3 沿列方向连接起来
        result = concat([df1, df3], axis=1)

        # 期望的日期时间索引 exp_idx，不带指定时区
        exp_idx = DatetimeIndex(
            [
                "2010-12-31 15:00:00+00:00",
                "2010-12-31 16:00:00+00:00",
                "2010-12-31 17:00:00+00:00",
                "2010-12-31 23:00:00+00:00",
                "2011-01-01 00:00:00+00:00",
                "2011-01-01 01:00:00+00:00",
            ]
        ).as_unit("ns")

        # 创建期望的 DataFrame，包含预期的数据和索引
        expected = DataFrame(
            [
                [np.nan, 1],
                [np.nan, 2],
                [np.nan, 3],
                [1, np.nan],
                [2, np.nan],
                [3, np.nan],
            ],
            index=exp_idx,
            columns=["a", "b"],
        )

        # 断言 result 和 expected 的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # GH 13783: 在重新采样后进行连接
        # 对 df1 和 df2 按小时重新采样并取均值后进行连接，同时按索引排序
        result = concat([df1.resample("h").mean(), df2.resample("h").mean()], sort=True)
        # 创建期望的 DataFrame，包含预期的数据和索引
        expected = DataFrame(
            {"a": [1, 2, 3] + [np.nan] * 3, "b": [np.nan] * 3 + [1, 2, 3]},
            index=idx1.append(idx1),
        )
        # 断言 result 和 expected 的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_concat_datetimeindex_freq(self):
        # GH 3232
        # 创建一个时间范围索引 dr，带有 50 毫秒的频率和 UTC 时区
        dr = date_range("01-Jan-2013", periods=100, freq="50ms", tz="UTC")
        # 创建包含数据的列表
        data = list(range(100))
        # 创建期望的 DataFrame，带有指定的数据和索引
        expected = DataFrame(data, index=dr)
        # 将 expected 的前半部分和后半部分连接起来
        result = concat([expected[:50], expected[50:]])
        # 断言 result 和 expected 的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 将 expected 的后半部分和前半部分连接起来
        result = concat([expected[50:], expected[:50]])
        # 创建期望的 DataFrame，带有指定的数据和索引
        expected = DataFrame(data[50:] + data[:50], index=dr[50:].append(dr[:50]))
        expected.index._data.freq = None  # 清除索引的频率信息
        # 断言 result 和 expected 的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
    def test_concat_multiindex_datetime_object_index(self):
        # 测试拼接多级索引和日期时间对象索引的情况
        idx = Index(
            [dt.date(2013, 1, 1), dt.date(2014, 1, 1), dt.date(2015, 1, 1)],
            dtype="object",
        )

        s = Series(
            ["a", "b"],
            index=MultiIndex.from_arrays(
                [
                    [1, 2],
                    idx[:-1],
                ],
                names=["first", "second"],
            ),
        )
        s2 = Series(
            ["a", "b"],
            index=MultiIndex.from_arrays(
                [[1, 2], idx[::2]],
                names=["first", "second"],
            ),
        )
        mi = MultiIndex.from_arrays(
            [[1, 2, 2], idx],
            names=["first", "second"],
        )
        assert mi.levels[1].dtype == object

        expected = DataFrame(
            [["a", "a"], ["b", np.nan], [np.nan, "b"]],
            index=mi,
        )
        result = concat([s, s2], axis=1)
        tm.assert_frame_equal(result, expected)

    def test_concat_NaT_series(self):
        # 测试合并 NaT 系列和日期时间系列的情况
        # GH 11693
        x = Series(
            date_range("20151124 08:00", "20151124 09:00", freq="1h", tz="US/Eastern")
        )
        y = Series(pd.NaT, index=[0, 1], dtype="datetime64[ns, US/Eastern]")
        expected = Series([x[0], x[1], pd.NaT, pd.NaT])

        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

        # 全部为 NaT 并且有时区信息
        expected = Series(pd.NaT, index=range(4), dtype="datetime64[ns, US/Eastern]")
        result = concat([y, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    def test_concat_NaT_series2(self):
        # 没有时区信息的情况
        x = Series(date_range("20151124 08:00", "20151124 09:00", freq="1h"))
        y = Series(date_range("20151124 10:00", "20151124 11:00", freq="1h"))
        y[:] = pd.NaT
        expected = Series([x[0], x[1], pd.NaT, pd.NaT])
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

        # 全部为 NaT 并且没有时区信息
        x[:] = pd.NaT
        expected = Series(pd.NaT, index=range(4), dtype="datetime64[ns]")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_concat_NaT_dataframes(self, tz):
        # GH 12396

        # 创建一个日期时间索引，包含两个 NaT 值，并指定时区
        dti = DatetimeIndex([pd.NaT, pd.NaT], tz=tz)
        # 创建第一个 DataFrame，包含一个列，列的值为上述日期时间索引
        first = DataFrame({0: dti})
        # 创建第二个 DataFrame，包含两行，每行包含一个带时区的 Timestamp 对象
        second = DataFrame(
            [[Timestamp("2015/01/01", tz=tz)], [Timestamp("2016/01/01", tz=tz)]],
            index=[2, 3],
        )
        # 创建预期的 DataFrame，包含四行，其中两行是 NaT，另外两行是带时区的 Timestamp 对象
        expected = DataFrame(
            [
                pd.NaT,
                pd.NaT,
                Timestamp("2015/01/01", tz=tz),
                Timestamp("2016/01/01", tz=tz),
            ]
        )

        # 进行 DataFrame 的纵向连接，结果保存在 result 中
        result = concat([first, second], axis=0)
        # 使用测试框架验证 result 是否与 expected 相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz1", [None, "UTC"])
    @pytest.mark.parametrize("tz2", [None, "UTC"])
    @pytest.mark.parametrize("item", [pd.NaT, Timestamp("20150101").as_unit("ns")])
    def test_concat_NaT_dataframes_all_NaT_axis_0(self, tz1, tz2, item):
        # GH 12396

        # tz-naive 情况下，创建第一个 DataFrame，其中包含两行 NaT 值并进行时区本地化处理
        first = DataFrame([[pd.NaT], [pd.NaT]]).apply(lambda x: x.dt.tz_localize(tz1))
        # 创建第二个 DataFrame，包含一个 item 值并进行时区本地化处理
        second = DataFrame([item]).apply(lambda x: x.dt.tz_localize(tz2))

        # 进行 DataFrame 的纵向连接，结果保存在 result 中
        result = concat([first, second], axis=0)
        # 创建预期的 DataFrame，包含三行，其中两行为 NaT，一行为带时区的 item 值
        expected = DataFrame(Series([pd.NaT, pd.NaT, item], index=[0, 1, 0]))
        # 对预期的 DataFrame 进行时区本地化处理
        expected = expected.apply(lambda x: x.dt.tz_localize(tz2))
        # 如果 tz1 和 tz2 不相同，则将预期结果转换为对象类型
        if tz1 != tz2:
            expected = expected.astype(object)

        # 使用测试框架验证 result 是否与 expected 相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz1", [None, "UTC"])
    @pytest.mark.parametrize("tz2", [None, "UTC"])
    def test_concat_NaT_dataframes_all_NaT_axis_1(self, tz1, tz2):
        # GH 12396

        # 创建第一个 DataFrame，其中包含两行 NaT 值并进行时区本地化处理
        first = DataFrame(Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1))
        # 创建第二个 DataFrame，包含一列带时区的 NaT 值
        second = DataFrame(Series([pd.NaT]).dt.tz_localize(tz2), columns=[1])
        # 创建预期的 DataFrame，包含两列，每列包含两行对应的带时区的 NaT 值
        expected = DataFrame(
            {
                0: Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1),
                1: Series([pd.NaT, pd.NaT]).dt.tz_localize(tz2),
            }
        )
        # 进行 DataFrame 的横向连接，结果保存在 result 中
        result = concat([first, second], axis=1)
        # 使用测试框架验证 result 是否与 expected 相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz1", [None, "UTC"])
    @pytest.mark.parametrize("tz2", [None, "UTC"])
    def test_concat_NaT_series_dataframe_all_NaT(self, tz1, tz2):
        # GH 12396

        # tz-naive 情况下，创建第一个 Series，其中包含两个 NaT 值并进行时区本地化处理
        first = Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1)
        # 创建第二个 DataFrame，包含两行，每行包含一个带时区的 Timestamp 对象
        second = DataFrame(
            [
                [Timestamp("2015/01/01", tz=tz2)],
                [Timestamp("2016/01/01", tz=tz2)],
            ],
            index=[2, 3],
        )

        # 创建预期的 DataFrame，包含四行，其中两行是 NaT，另外两行是带时区的 Timestamp 对象
        expected = DataFrame(
            [
                pd.NaT,
                pd.NaT,
                Timestamp("2015/01/01", tz=tz2),
                Timestamp("2016/01/01", tz=tz2),
            ]
        )
        # 如果 tz1 和 tz2 不相同，则将预期结果转换为对象类型
        if tz1 != tz2:
            expected = expected.astype(object)

        # 进行 Series 和 DataFrame 的连接，结果保存在 result 中
        result = concat([first, second])
        # 使用测试框架验证 result 是否与 expected 相等
        tm.assert_frame_equal(result, expected)
class TestTimezoneConcat:
    def test_concat_tz_series(self):
        # gh-11755: tz and no tz
        # 创建包含时区信息的时间序列 x
        x = Series(date_range("20151124 08:00", "20151124 09:00", freq="1h", tz="UTC"))
        # 创建不含时区信息的时间序列 y
        y = Series(date_range("2012-01-01", "2012-01-02"))
        # 期望的结果序列，包含 x 和 y 的前两个元素
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        # 进行序列的连接操作，忽略索引
        result = concat([x, y], ignore_index=True)
        # 断言连接结果与期望结果相等
        tm.assert_series_equal(result, expected)

    def test_concat_tz_series2(self):
        # gh-11887: concat tz and object
        # 创建包含时区信息的时间序列 x
        x = Series(date_range("20151124 08:00", "20151124 09:00", freq="1h", tz="UTC"))
        # 创建包含对象的时间序列 y
        y = Series(["a", "b"])
        # 期望的结果序列，包含 x 和 y 的前两个元素
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        # 进行序列的连接操作，忽略索引
        result = concat([x, y], ignore_index=True)
        # 断言连接结果与期望结果相等
        tm.assert_series_equal(result, expected)

    def test_concat_tz_series3(self, unit, unit2):
        # see gh-12217 and gh-12306
        # Concatenating two UTC times
        # 创建包含 UTC 时间的 DataFrame first
        first = DataFrame([[datetime(2016, 1, 1)]], dtype=f"M8[{unit}]")
        # 将第一列日期时间列设定为 UTC 时区
        first[0] = first[0].dt.tz_localize("UTC")

        # 创建包含 UTC 时间的 DataFrame second
        second = DataFrame([[datetime(2016, 1, 2)]], dtype=f"M8[{unit2}]")
        # 将第一列日期时间列设定为 UTC 时区
        second[0] = second[0].dt.tz_localize("UTC")

        # 进行 DataFrame 的连接操作
        result = concat([first, second])
        # 获取最精细时间单位
        exp_unit = tm.get_finest_unit(unit, unit2)
        # 断言连接后的第一列数据类型符合预期
        assert result[0].dtype == f"datetime64[{exp_unit}, UTC]"

    def test_concat_tz_series4(self, unit, unit2):
        # Concatenating two London times
        # 创建包含伦敦时间的 DataFrame first
        first = DataFrame([[datetime(2016, 1, 1)]], dtype=f"M8[{unit}]")
        # 将第一列日期时间列设定为 Europe/London 时区
        first[0] = first[0].dt.tz_localize("Europe/London")

        # 创建包含伦敦时间的 DataFrame second
        second = DataFrame([[datetime(2016, 1, 2)]], dtype=f"M8[{unit2}]")
        # 将第一列日期时间列设定为 Europe/London 时区
        second[0] = second[0].dt.tz_localize("Europe/London")

        # 进行 DataFrame 的连接操作
        result = concat([first, second])
        # 获取最精细时间单位
        exp_unit = tm.get_finest_unit(unit, unit2)
        # 断言连接后的第一列数据类型符合预期
        assert result[0].dtype == f"datetime64[{exp_unit}, Europe/London]"

    def test_concat_tz_series5(self, unit, unit2):
        # Concatenating 2+1 London times
        # 创建包含伦敦时间的 DataFrame first
        first = DataFrame(
            [[datetime(2016, 1, 1)], [datetime(2016, 1, 2)]], dtype=f"M8[{unit}]"
        )
        # 将第一列日期时间列设定为 Europe/London 时区
        first[0] = first[0].dt.tz_localize("Europe/London")

        # 创建包含伦敦时间的 DataFrame second
        second = DataFrame([[datetime(2016, 1, 3)]], dtype=f"M8[{unit2}]")
        # 将第一列日期时间列设定为 Europe/London 时区
        second[0] = second[0].dt.tz_localize("Europe/London")

        # 进行 DataFrame 的连接操作
        result = concat([first, second])
        # 获取最精细时间单位
        exp_unit = tm.get_finest_unit(unit, unit2)
        # 断言连接后的第一列数据类型符合预期
        assert result[0].dtype == f"datetime64[{exp_unit}, Europe/London]"
    # 定义一个测试方法，用于测试时区序列的连接操作，接受两个时区单位作为参数
    def test_concat_tz_series6(self, unit, unit2):
        # 创建包含一个日期时间对象的 DataFrame，指定日期时间单位为 unit
        first = DataFrame([[datetime(2016, 1, 1)]], dtype=f"M8[{unit}]")
        # 将第一个日期时间列本地化为"Europe/London"时区
        first[0] = first[0].dt.tz_localize("Europe/London")

        # 创建包含两个日期时间对象的 DataFrame，指定日期时间单位为 unit2
        second = DataFrame(
            [[datetime(2016, 1, 2)], [datetime(2016, 1, 3)]], dtype=f"M8[{unit2}]"
        )
        # 将第一个日期时间列本地化为"Europe/London"时区
        second[0] = second[0].dt.tz_localize("Europe/London")

        # 将两个 DataFrame 连接起来
        result = concat([first, second])
        # 获取最细粒度的时间单位
        exp_unit = tm.get_finest_unit(unit, unit2)
        # 断言第一个结果列的数据类型为本地化后的日期时间类型，包含 Europe/London 时区信息
        assert result[0].dtype == f"datetime64[{exp_unit}, Europe/London]"

    # 定义一个测试方法，测试带有本地时区的日期时间序列连接操作
    def test_concat_tz_series_tzlocal(self):
        # 创建包含本地时区的时间戳对象列表
        x = [
            Timestamp("2011-01-01", tz=dateutil.tz.tzlocal()),
            Timestamp("2011-02-01", tz=dateutil.tz.tzlocal()),
        ]
        # 创建包含本地时区的时间戳对象列表
        y = [
            Timestamp("2012-01-01", tz=dateutil.tz.tzlocal()),
            Timestamp("2012-02-01", tz=dateutil.tz.tzlocal()),
        ]

        # 将两个 Series 连接起来，忽略索引
        result = concat([Series(x), Series(y)], ignore_index=True)
        # 断言结果 Series 的数据类型为本地时区的日期时间类型
        tm.assert_series_equal(result, Series(x + y))
        assert result.dtype == "datetime64[s, tzlocal()]"

    # 定义一个测试方法，测试带有日期时间样式的日期时间序列连接操作
    def test_concat_tz_series_with_datetimelike(self):
        # 创建包含时区信息的时间戳对象列表
        x = [
            Timestamp("2011-01-01", tz="US/Eastern"),
            Timestamp("2011-02-01", tz="US/Eastern"),
        ]
        # 创建包含 Timedelta 对象的列表
        y = [pd.Timedelta("1 day"), pd.Timedelta("2 day")]
        # 将两个 Series 连接起来，忽略索引
        result = concat([Series(x), Series(y)], ignore_index=True)
        # 断言结果 Series 的数据类型为对象类型
        tm.assert_series_equal(result, Series(x + y, dtype="object"))

        # 创建包含 Period 对象的列表
        y = [pd.Period("2011-03", freq="M"), pd.Period("2011-04", freq="M")]
        # 将两个 Series 连接起来，忽略索引
        result = concat([Series(x), Series(y)], ignore_index=True)
        # 断言结果 Series 的数据类型为对象类型
        tm.assert_series_equal(result, Series(x + y, dtype="object"))

    # 定义一个测试方法，测试带有时区信息的 DataFrame 连接操作
    def test_concat_tz_frame(self):
        # 创建包含时区信息的 DataFrame
        df2 = DataFrame(
            {
                "A": Timestamp("20130102", tz="US/Eastern"),
                "B": Timestamp("20130603", tz="CET"),
            },
            index=range(5),
        )

        # 将 DataFrame df2 的列 A 和 B 连接起来形成新的 DataFrame df3
        df3 = concat([df2.A.to_frame(), df2.B.to_frame()], axis=1)
        # 断言 DataFrame df2 和 df3 相等
        tm.assert_frame_equal(df2, df3)
    def test_concat_multiple_tzs(self):
        # GH#12467
        # combining datetime tz-aware and naive DataFrames
        # 创建不同时区感知和非感知的时间戳
        ts1 = Timestamp("2015-01-01", tz=None)
        ts2 = Timestamp("2015-01-01", tz="UTC")
        ts3 = Timestamp("2015-01-01", tz="EST")

        # 创建包含时间戳的数据框架
        df1 = DataFrame({"time": [ts1]})
        df2 = DataFrame({"time": [ts2]})
        df3 = DataFrame({"time": [ts3]})

        # 连接数据框架并重置索引
        results = concat([df1, df2]).reset_index(drop=True)
        expected = DataFrame({"time": [ts1, ts2]}, dtype=object)
        tm.assert_frame_equal(results, expected)

        results = concat([df1, df3]).reset_index(drop=True)
        expected = DataFrame({"time": [ts1, ts3]}, dtype=object)
        tm.assert_frame_equal(results, expected)

        results = concat([df2, df3]).reset_index(drop=True)
        expected = DataFrame({"time": [ts2, ts3]})
        tm.assert_frame_equal(results, expected)

    def test_concat_multiindex_with_tz(self):
        # GH 6606
        # 创建具有时区的多级索引数据框架
        df = DataFrame(
            {
                "dt": DatetimeIndex(
                    [
                        datetime(2014, 1, 1),
                        datetime(2014, 1, 2),
                        datetime(2014, 1, 3),
                    ],
                    dtype="M8[ns, US/Pacific]",
                ),
                "b": ["A", "B", "C"],
                "c": [1, 2, 3],
                "d": [4, 5, 6],
            }
        )
        df = df.set_index(["dt", "b"])

        # 创建预期的多级索引
        exp_idx1 = DatetimeIndex(
            ["2014-01-01", "2014-01-02", "2014-01-03"] * 2,
            dtype="M8[ns, US/Pacific]",
            name="dt",
        )
        exp_idx2 = Index(["A", "B", "C"] * 2, name="b")
        exp_idx = MultiIndex.from_arrays([exp_idx1, exp_idx2])
        expected = DataFrame(
            {"c": [1, 2, 3] * 2, "d": [4, 5, 6] * 2}, index=exp_idx, columns=["c", "d"]
        )

        result = concat([df, df])
        tm.assert_frame_equal(result, expected)

    def test_concat_tz_not_aligned(self):
        # GH#22796
        # 创建带有不对齐时区的数据框架
        ts = pd.to_datetime([1, 2]).tz_localize("UTC")
        a = DataFrame({"A": ts})
        b = DataFrame({"A": ts, "B": ts})

        # 连接数据框架并设置参数进行比较
        result = concat([a, b], sort=True, ignore_index=True)
        expected = DataFrame(
            {"A": list(ts) + list(ts), "B": [pd.NaT, pd.NaT] + list(ts)}
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "t1",
        [
            "2015-01-01",
            pytest.param(
                pd.NaT,
                marks=pytest.mark.xfail(
                    reason="GH23037 incorrect dtype when concatenating"
                ),
            ),
        ],
    )
    # 定义一个测试方法，用于测试合并带时区信息的多列 DataFrame
    def test_concat_tz_NaT(self, t1):
        # GH#22796
        # 在这个测试中，我们测试合并带时区信息的多列 DataFrame
        # 创建第一个带时区信息的时间戳对象 ts1，使用给定的时间 t1 和 UTC 时区
        ts1 = Timestamp(t1, tz="UTC")
        # 创建第二个和第三个带时区信息的时间戳对象，日期均为 "2015-01-01"，时区为 UTC
        ts2 = Timestamp("2015-01-01", tz="UTC")
        ts3 = Timestamp("2015-01-01", tz="UTC")

        # 创建第一个 DataFrame df1，包含一个列表，列表中有两个时间戳对象 ts1 和 ts2
        df1 = DataFrame([[ts1, ts2]])
        # 创建第二个 DataFrame df2，包含一个列表，列表中有一个时间戳对象 ts3
        df2 = DataFrame([[ts3]])

        # 执行 concat 函数，将 df1 和 df2 合并
        result = concat([df1, df2])
        # 创建预期的 DataFrame expected，包含两行，第一行与 df1 相同，第二行的第二列值为 pd.NaT
        expected = DataFrame([[ts1, ts2], [ts3, pd.NaT]], index=[0, 0])

        # 使用 assert_frame_equal 函数断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义另一个测试方法，用于测试带时区信息和空 DataFrame 的合并
    def test_concat_tz_with_empty(self):
        # GH 9188
        # 在这个测试中，我们测试带时区信息的 DataFrame 与空 DataFrame 的合并
        # 使用 date_range 函数创建一个带时区信息的 DataFrame，起始日期为 "2000"，仅包含一个日期，时区为 UTC
        result = concat(
            [DataFrame(date_range("2000", periods=1, tz="UTC")), DataFrame()]
        )
        # 创建预期的 DataFrame expected，包含一个带时区信息的日期范围，起始日期为 "2000"，时区为 UTC
        expected = DataFrame(date_range("2000", periods=1, tz="UTC"))

        # 使用 assert_frame_equal 函数断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
class TestPeriodConcat:
    def test_concat_period_series(self):
        # 创建包含日期周期的 Series x 和 y
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(pd.PeriodIndex(["2015-10-01", "2016-01-01"], freq="D"))
        # 期望结果是合并后的 Series，包含所有元素，数据类型为 'Period[D]'
        expected = Series([x[0], x[1], y[0], y[1]], dtype="Period[D]")
        # 使用 concat 函数合并 x 和 y，忽略索引
        result = concat([x, y], ignore_index=True)
        # 断言合并结果与期望结果相等
        tm.assert_series_equal(result, expected)

    def test_concat_period_multiple_freq_series(self):
        # 创建包含不同频率日期周期的 Series x 和 y
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(pd.PeriodIndex(["2015-10-01", "2016-01-01"], freq="M"))
        # 期望结果是合并后的 Series，数据类型为 'object'
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        # 使用 concat 函数合并 x 和 y，忽略索引
        result = concat([x, y], ignore_index=True)
        # 断言合并结果与期望结果相等
        tm.assert_series_equal(result, expected)
        # 进行额外的断言，验证合并后的 Series 数据类型为 'object'
        assert result.dtype == "object"

    def test_concat_period_other_series(self):
        # 创建包含不同频率日期周期的 Series x 和 y，但都是 'object'
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="M"))
        # 期望结果是合并后的 Series，数据类型为 'object'
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        # 使用 concat 函数合并 x 和 y，忽略索引
        result = concat([x, y], ignore_index=True)
        # 断言合并结果与期望结果相等
        tm.assert_series_equal(result, expected)
        # 进行额外的断言，验证合并后的 Series 数据类型为 'object'
        assert result.dtype == "object"

    def test_concat_period_other_series2(self):
        # 创建包含日期周期和日期时间索引的 Series x 和 y
        # 针对 y，注释指出它是非周期数据
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(DatetimeIndex(["2015-11-01", "2015-12-01"]))
        # 期望结果是合并后的 Series，数据类型为 'object'
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        # 使用 concat 函数合并 x 和 y，忽略索引
        result = concat([x, y], ignore_index=True)
        # 断言合并结果与期望结果相等
        tm.assert_series_equal(result, expected)
        # 进行额外的断言，验证合并后的 Series 数据类型为 'object'
        assert result.dtype == "object"

    def test_concat_period_other_series3(self):
        # 创建包含日期周期和字符串索引的 Series x 和 y
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(["A", "B"])
        # 期望结果是合并后的 Series，数据类型为 'object'
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        # 使用 concat 函数合并 x 和 y，忽略索引
        result = concat([x, y], ignore_index=True)
        # 断言合并结果与期望结果相等
        tm.assert_series_equal(result, expected)
        # 进行额外的断言，验证合并后的 Series 数据类型为 'object'
        assert result.dtype == "object"


def test_concat_timedelta64_block():
    # 创建时间增量的 Series
    rng = to_timedelta(np.arange(10), unit="s")
    df = DataFrame({"time": rng})
    # 使用 concat 函数合并 df 和 df，没有指定轴，默认按行合并
    result = concat([df, df])
    # 断言前 10 行的合并结果与原始 df 相等
    tm.assert_frame_equal(result.iloc[:10], df)
    # 断言后 10 行的合并结果与原始 df 相等
    tm.assert_frame_equal(result.iloc[10:], df)


def test_concat_multiindex_datetime_nat():
    # 创建包含 MultiIndex 和 NaT 的 DataFrame left 和 right
    left = DataFrame({"a": 1}, index=MultiIndex.from_tuples([(1, pd.NaT)]))
    right = DataFrame(
        {"b": 2}, index=MultiIndex.from_tuples([(1, pd.NaT), (2, pd.NaT)])
    )
    # 使用 concat 函数按列合并 left 和 right
    result = concat([left, right], axis="columns")
    # 创建期望的 DataFrame，包含合并后的数据，索引为 MultiIndex
    expected = DataFrame(
        {"a": [1.0, np.nan], "b": 2}, MultiIndex.from_tuples([(1, pd.NaT), (2, pd.NaT)])
    )
    # 断言合并结果与期望结果相等
    tm.assert_frame_equal(result, expected)


def test_concat_float_datetime64():
    # 创建包含日期时间和浮点数的 DataFrame
    df_time = DataFrame({"A": pd.array(["2000"], dtype="datetime64[ns]")})
    df_float = DataFrame({"A": pd.array([1.0], dtype="float64")})
    # 创建预期的 DataFrame，包含列"A"，第一个元素是 datetime64 类型的时间戳 "2000"，第二个元素是 float64 类型的数值 1.0
    expected = DataFrame(
        {
            "A": [
                pd.array(["2000"], dtype="datetime64[ns]")[0],
                pd.array([1.0], dtype="float64")[0],
            ]
        },
        index=[0, 0],
    )
    # 将 df_time 和 df_float 进行合并
    result = concat([df_time, df_float])
    # 使用测试框架中的函数检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)

    # 创建预期的 DataFrame，只包含列"A"，数据类型为 object 的空数组
    expected = DataFrame({"A": pd.array([], dtype="object")})
    # 对 df_time 和 df_float 的空切片进行合并
    result = concat([df_time.iloc[:0], df_float.iloc[:0]])
    # 使用测试框架中的函数检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)

    # 创建预期的 DataFrame，只包含列"A"，数据类型为 object 的数组，包含一个元素 1.0
    expected = DataFrame({"A": pd.array([1.0], dtype="object")})
    # 对 df_time 的空切片和 df_float 的合并
    result = concat([df_time.iloc[:0], df_float])
    # 使用测试框架中的函数检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)

    # 创建预期的 DataFrame，只包含列"A"，数据类型为 datetime64[ns] 的时间戳 "2000"，转换为 object 类型
    expected = DataFrame({"A": pd.array(["2000"], dtype="datetime64[ns]")}).astype(
        object
    )
    # 对 df_time 和 df_float 的空切片进行合并
    result = concat([df_time, df_float.iloc[:0]])
    # 使用测试框架中的函数检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
```