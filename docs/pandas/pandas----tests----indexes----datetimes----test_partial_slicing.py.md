# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_partial_slicing.py`

```
# 导入所需的模块和库
from datetime import datetime

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 库

from pandas import (  # 从 pandas 中导入多个模块和类
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm  # 导入 pandas 测试模块

# 定义测试类 TestSlicing
class TestSlicing:
    
    # 测试方法：测试索引为字符串时 Series 名称的转换
    def test_string_index_series_name_converted(self):
        # 创建一个 DataFrame，包含随机数据，使用日期范围作为索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            index=date_range("1/1/2000", periods=10),
        )

        # 通过 loc 方法获取特定日期的数据
        result = df.loc["1/3/2000"]
        # 断言结果的名称与索引第三个元素的名称相同
        assert result.name == df.index[2]

        # 对 DataFrame 进行转置后，再次使用 loc 方法获取特定日期的数据
        result = df.T["1/3/2000"]
        # 断言结果的名称与索引第三个元素的名称相同
        assert result.name == df.index[2]

    # 测试方法：测试带时区信息的字符串化切片
    def test_stringified_slice_with_tz(self):
        # 定义起始日期
        start = "2013-01-07"
        # 创建带有时区信息的日期范围索引
        idx = date_range(start=start, freq="1d", periods=10, tz="US/Eastern")
        # 创建 DataFrame，索引为上述创建的日期范围索引，数据为0到9
        df = DataFrame(np.arange(10), index=idx)
        # 进行切片操作，不会引发异常

    # 测试方法：测试返回类型不依赖于单调性
    def test_return_type_doesnt_depend_on_monotonicity(self):
        # 创建分钟级频率的日期范围索引，从指定日期开始
        dti = date_range(start="2015-5-13 23:59:00", freq="min", periods=3)
        # 创建 Series，使用上述日期范围索引和对应的值
        ser = Series(range(3), index=dti)

        # 创建非单调索引的 Series
        ser2 = Series(range(3), index=[dti[1], dti[0], dti[2]])

        # 定义低于分钟级别的键值
        key = "2015-5-14 00"

        # 对单调递增索引进行 loc 操作
        result = ser.loc[key]
        # 断言结果与预期的单调递增索引切片相等
        expected = ser.iloc[1:]
        tm.assert_series_equal(result, expected)

        # 对单调递减索引进行 loc 操作
        result = ser.iloc[::-1].loc[key]
        # 断言结果与预期的单调递减索引切片相等
        expected = ser.iloc[::-1][:-1]
        tm.assert_series_equal(result, expected)

        # 对非单调索引进行 loc 操作
        result2 = ser2.loc[key]
        # 断言结果与预期的非单调索引切片相等
        expected2 = ser2.iloc[::2]
        tm.assert_series_equal(result2, expected2)

    # 测试方法：测试返回类型不依赖于单调性（更高分辨率）
    def test_return_type_doesnt_depend_on_monotonicity_higher_reso(self):
        # 创建分钟级频率的日期范围索引，从指定日期开始
        dti = date_range(start="2015-5-13 23:59:00", freq="min", periods=3)
        # 创建 Series，使用上述日期范围索引和对应的值
        ser = Series(range(3), index=dti)

        # 创建非单调索引的 Series
        ser2 = Series(range(3), index=[dti[1], dti[0], dti[2]])

        # 定义高于分钟级别的键值
        key = "2015-5-14 00:00:00"

        # 对单调递增索引进行 loc 操作
        result = ser.loc[key]
        # 断言结果与预期相等
        assert result == 1

        # 对单调递减索引进行 loc 操作
        result = ser.iloc[::-1].loc[key]
        # 断言结果与预期相等
        assert result == 1

        # 对非单调索引进行 loc 操作
        result2 = ser2.loc[key]
        # 断言结果与预期相等
        assert result2 == 0
    def test_monotone_DTI_indexing_bug(self):
        # GH 19362
        # Testing accessing the first element in a monotonic descending
        # partial string indexing.

        # 创建一个包含 0 到 4 的 DataFrame 对象
        df = DataFrame(list(range(5)))
        # 创建一个日期字符串列表
        date_list = [
            "2018-01-02",
            "2017-02-10",
            "2016-03-10",
            "2015-03-15",
            "2014-03-16",
        ]
        # 使用日期字符串列表创建日期索引对象
        date_index = DatetimeIndex(date_list)
        # 在 DataFrame 中增加一个名为 'date' 的列，并赋值为日期索引对象
        df["date"] = date_index
        # 创建预期结果的 DataFrame 对象
        expected = DataFrame({0: list(range(5)), "date": date_index})
        # 使用测试工具验证 df 和预期结果是否相等
        tm.assert_frame_equal(df, expected)

        # 因为 df.index 的分辨率是每小时，而我们使用每日分辨率的字符串进行切片，所以我们得到一个切片。
        # 如果两者都是每日分辨率，我们将只获得一个单独的项。
        dti = date_range("20170101 01:00:00", periods=3)
        # 创建一个新的 DataFrame 对象，具有逆序索引和单列 'A'，并使用 dti 作为索引
        df = DataFrame({"A": [1, 2, 3]}, index=dti[::-1])

        # 创建预期结果的 DataFrame 对象，具有值为 1 的单列 'A'，并使用逆序的 dti[-1:] 作为索引
        expected = DataFrame({"A": 1}, index=dti[-1:][::-1])
        # 使用 loc 方法获取索引为 "2017-01-03" 的结果
        result = df.loc["2017-01-03"]
        # 使用测试工具验证 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 创建逆序的 DataFrame 对象，然后使用 loc 方法获取索引为 "2017-01-03" 的结果
        result2 = df.iloc[::-1].loc["2017-01-03"]
        # 创建预期结果的逆序 DataFrame 对象
        expected2 = expected.iloc[::-1]
        # 使用测试工具验证 result2 和 expected2 是否相等
        tm.assert_frame_equal(result2, expected2)

    def test_slice_year(self):
        # 创建一个包含每个工作日的日期索引对象，从 2005 年 1 月 1 日开始，共 500 个工作日
        dti = date_range(freq="B", start=datetime(2005, 1, 1), periods=500)

        # 创建一个 Series 对象，其值为其索引的位置，索引为 dti
        s = Series(np.arange(len(dti)), index=dti)
        # 使用字符串 "2005" 对 s 进行切片
        result = s["2005"]
        # 创建预期结果的 Series 对象，其索引为 2005 年的索引
        expected = s[s.index.year == 2005]
        # 使用测试工具验证 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个 DataFrame 对象，其值为随机数，索引为 dti
        df = DataFrame(np.random.default_rng(2).random((len(dti), 5)), index=dti)
        # 使用字符串 "2005" 对 df 进行切片
        result = df.loc["2005"]
        # 创建预期结果的 DataFrame 对象，其索引为 2005 年的索引
        expected = df[df.index.year == 2005]
        # 使用测试工具验证 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "partial_dtime",
        [
            "2019",
            "2019Q4",
            "Dec 2019",
            "2019-12-31",
            "2019-12-31 23",
            "2019-12-31 23:59",
        ],
    )
    def test_slice_end_of_period_resolution(self, partial_dtime):
        # GH#31064
        # 创建一个包含 10 个时间戳的日期索引对象，从 "2019-12-31 23:59:55.999999999" 开始，频率为每秒
        dti = date_range("2019-12-31 23:59:55.999999999", periods=10, freq="s")

        # 创建一个 Series 对象，其值为其索引的位置，索引为 dti
        ser = Series(range(10), index=dti)
        # 使用 partial_dtime 对 ser 进行切片
        result = ser[partial_dtime]
        # 创建预期结果的 Series 对象，其值为前五个位置的数据
        expected = ser.iloc[:5]
        # 使用测试工具验证 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    def test_slice_quarter(self):
        # 创建一个包含 500 天的日期索引对象，从 2000 年 6 月 1 日开始，频率为每天
        dti = date_range(freq="D", start=datetime(2000, 6, 1), periods=500)

        # 创建一个 Series 对象，其值为其索引的位置，索引为 dti
        s = Series(np.arange(len(dti)), index=dti)
        # 断言对字符串 "2001Q1" 进行切片后返回的长度为 90
        assert len(s["2001Q1"]) == 90

        # 创建一个 DataFrame 对象，其值为随机数，索引为 dti
        df = DataFrame(np.random.default_rng(2).random((len(dti), 5)), index=dti)
        # 断言对字符串 "1Q01" 进行切片后返回的长度为 90
        assert len(df.loc["1Q01"]) == 90

    def test_slice_month(self):
        # 创建一个包含 500 天的日期索引对象，从 2005 年 1 月 1 日开始，频率为每天
        dti = date_range(freq="D", start=datetime(2005, 1, 1), periods=500)
        # 创建一个 Series 对象，其值为其索引的位置，索引为 dti
        s = Series(np.arange(len(dti)), index=dti)
        # 断言对字符串 "2005-11" 进行切片后返回的长度为 30
        assert len(s["2005-11"]) == 30

        # 创建一个 DataFrame 对象，其值为随机数，索引为 dti
        df = DataFrame(np.random.default_rng(2).random((len(dti), 5)), index=dti)
        # 断言对字符串 "2005-11" 进行切片后返回的长度为 30
        assert len(df.loc["2005-11"]) == 30

        # 使用测试工具验证 s["2005-11"] 和 s["11-2005"] 是否相等
        tm.assert_series_equal(s["2005-11"], s["11-2005"])
    def test_partial_slice(self):
        # 创建一个日期范围，从2005年1月1日开始，每日频率，共500个时间点
        rng = date_range(freq="D", start=datetime(2005, 1, 1), periods=500)
        # 创建一个序列，其中数据是按照日期范围的索引排列
        s = Series(np.arange(len(rng)), index=rng)

        # 对序列进行部分切片，选择2005年5月到2006年2月之间的数据
        result = s["2005-05":"2006-02"]
        # 期望的结果是20050501到20060228之间的数据
        expected = s["20050501":"20060228"]
        # 断言切片后的结果与期望的结果相等
        tm.assert_series_equal(result, expected)

        # 对序列进行部分切片，选择2005年5月1日之后的所有数据
        result = s["2005-05":]
        # 期望的结果是从20050501之后的所有数据
        expected = s["20050501":]
        # 断言切片后的结果与期望的结果相等
        tm.assert_series_equal(result, expected)

        # 对序列进行部分切片，选择2006年2月之前的所有数据
        result = s[:"2006-02"]
        # 期望的结果是到20060228为止的所有数据
        expected = s[:"20060228"]
        # 断言切片后的结果与期望的结果相等
        tm.assert_series_equal(result, expected)

        # 对序列进行单独日期的切片，选择2005年1月1日的数据
        result = s["2005-1-1"]
        # 断言切片后的结果与序列中第一个数据相等
        assert result == s.iloc[0]

        # 使用 pytest 检查如果尝试访问不存在的日期（2004年12月31日），会引发 KeyError 错误并匹配给定的正则表达式消息
        with pytest.raises(KeyError, match=r"^'2004-12-31'$"):
            s["2004-12-31"]

    def test_partial_slice_daily(self):
        # 创建一个日期范围，从2005年1月31日开始，每小时频率，共500个时间点
        rng = date_range(freq="h", start=datetime(2005, 1, 31), periods=500)
        # 创建一个序列，其中数据是按照日期范围的索引排列
        s = Series(np.arange(len(rng)), index=rng)

        # 对序列进行部分切片，选择2005年1月31日当天的所有数据
        result = s["2005-1-31"]
        # 断言切片后的结果与序列中第一个到第24个小时的数据相等
        tm.assert_series_equal(result, s.iloc[:24])

        # 使用 pytest 检查如果尝试访问不存在的日期时间点（2004年12月31日 00时），会引发 KeyError 错误并匹配给定的正则表达式消息
        with pytest.raises(KeyError, match=r"^'2004-12-31 00'$"):
            s["2004-12-31 00"]

    def test_partial_slice_hourly(self):
        # 创建一个日期范围，从2005年1月1日 20:00开始，每分钟频率，共500个时间点
        rng = date_range(freq="min", start=datetime(2005, 1, 1, 20, 0, 0), periods=500)
        # 创建一个序列，其中数据是按照日期范围的索引排列
        s = Series(np.arange(len(rng)), index=rng)

        # 对序列进行部分切片，选择2005年1月1日当天的所有数据
        result = s["2005-1-1"]
        # 断言切片后的结果与序列中第一个到第60分钟的数据相等
        tm.assert_series_equal(result, s.iloc[: 60 * 4])

        # 对序列进行部分切片，选择2005年1月1日 20:00到20:59之间的数据
        result = s["2005-1-1 20"]
        # 断言切片后的结果与序列中第一个到第60分钟的数据相等
        tm.assert_series_equal(result, s.iloc[:60])

        # 使用 assert 检查切片后的结果与序列中第一个数据（2005年1月1日 00:00）相等
        assert s["2005-1-1 20:00"] == s.iloc[0]
        # 使用 pytest 检查如果尝试访问不存在的日期时间点（2004年12月31日 00:15），会引发 KeyError 错误并匹配给定的正则表达式消息
        with pytest.raises(KeyError, match=r"^'2004-12-31 00:15'$"):
            s["2004-12-31 00:15"]

    def test_partial_slice_minutely(self):
        # 创建一个日期范围，从2005年1月1日 23:59:00开始，每秒频率，共500个时间点
        rng = date_range(freq="s", start=datetime(2005, 1, 1, 23, 59, 0), periods=500)
        # 创建一个序列，其中数据是按照日期范围的索引排列
        s = Series(np.arange(len(rng)), index=rng)

        # 对序列进行部分切片，选择2005年1月1日 23:59的数据
        result = s["2005-1-1 23:59"]
        # 断言切片后的结果与序列中第一个到第60秒的数据相等
        tm.assert_series_equal(result, s.iloc[:60])

        # 对序列进行部分切片，选择2005年1月1日当天的所有数据
        result = s["2005-1-1"]
        # 断言切片后的结果与序列中第一个到第60秒的数据相等
        tm.assert_series_equal(result, s.iloc[:60])

        # 使用 assert 检查切片后的结果与序列中第一个数据（2005年1月1日 23:59:00）相等
        assert s[Timestamp("2005-1-1 23:59:00")] == s.iloc[0]
        # 使用 pytest 检查如果尝试访问不存在的日期时间点（2004年12月31日 00:00:00），会引发 KeyError 错误并匹配给定的正则表达式消息
        with pytest.raises(KeyError, match=r"^'2004-12-31 00:00:00'$"):
            s["2004-12-31 00:00:00"]

    def test_partial_slice_second_precision(self):
        # 创建一个日期范围，从2005年1月1日 00:00:59.999990开始，每微秒频率，共20个时间点
        rng = date_range(
            start=datetime(2005, 1, 1, 0, 0, 59, microsecond=999990),
            periods=20,
            freq="us",
        )
        # 创建一个序列，其中数据是按照日期范围的索引排列
        s = Series(np.arange(20), rng)

        # 断言切片后的结果与序列中第一个到第10个微秒的数据相等
        tm.assert_series_equal(s["2005-1-1 00:00"], s.iloc[:10])
        # 断言切片后的结果与序列中第一个到第10个微秒的数据相等
        tm.assert_series_equal(s["2005-1-1 00:00:59"], s.iloc[:10])

        # 断言切片后的结果与序列中第11到第20个微秒的数据相等
        tm.assert_series_equal(s["2005-1-1 00:01"], s.iloc[10:])
        # 断言切片后的结果与序列中第11到第20个微秒的数据相等
        tm.assert_series_equal(s["2005-1-1 00:01:00"], s.iloc[10:])

        # 使用 assert 检查切片后的结果与序列中第一个数据（2005年1月1日 00:00:59
    def test_partial_slicing_with_multiindex(self):
        # GH 4758
        # 在多级索引中进行部分字符串切片索引存在问题
        df = DataFrame(
            {
                "ACCOUNT": ["ACCT1", "ACCT1", "ACCT1", "ACCT2"],
                "TICKER": ["ABC", "MNP", "XYZ", "XYZ"],
                "val": [1, 2, 3, 4],
            },
            index=date_range("2013-06-19 09:30:00", periods=4, freq="5min"),
        )
        # 将 DataFrame 设置为多级索引，附加在 ["ACCOUNT", "TICKER"] 上
        df_multi = df.set_index(["ACCOUNT", "TICKER"], append=True)

        # 预期结果为一个 DataFrame，包含值为 1 的单元格，索引为 ["ABC"]，列名为 ["val"]
        expected = DataFrame(
            [[1]], index=Index(["ABC"], name="TICKER"), columns=["val"]
        )
        # 使用 loc 方法从 df_multi 中获取索引为 ("2013-06-19 09:30:00", "ACCT1") 的结果
        result = df_multi.loc[("2013-06-19 09:30:00", "ACCT1")]
        # 使用测试模块的 assert_frame_equal 方法比较 result 和 expected
        tm.assert_frame_equal(result, expected)

        # 从 df_multi 中获取索引为 ("2013-06-19 09:30:00", "ACCT1", "ABC") 的预期结果
        expected = df_multi.loc[
            (Timestamp("2013-06-19 09:30:00", tz=None), "ACCT1", "ABC")
        ]
        # 使用 loc 方法获取索引为 ("2013-06-19 09:30:00", "ACCT1", "ABC") 的实际结果
        result = df_multi.loc[("2013-06-19 09:30:00", "ACCT1", "ABC")]
        # 使用测试模块的 assert_series_equal 方法比较 result 和 expected
        tm.assert_series_equal(result, expected)

        # 在第一级别进行部分字符串索引，同时在其他两个级别上进行标量索引
        result = df_multi.loc[("2013-06-19", "ACCT1", "ABC")]
        # 预期结果为 df_multi 的前一行，删除第一级和第二级索引
        expected = df_multi.iloc[:1].droplevel([1, 2])
        # 使用测试模块的 assert_frame_equal 方法比较 result 和 expected
        tm.assert_frame_equal(result, expected)

    def test_partial_slicing_with_multiindex_series(self):
        # GH 4294
        # 在系列多级索引上进行部分切片
        ser = Series(
            range(250),
            index=MultiIndex.from_product(
                [date_range("2000-1-1", periods=50), range(5)]
            ),
        )

        # 复制 ser 的前一部分，结果为 s2
        s2 = ser[:-1].copy()
        # 预期结果为 s2 中索引为 "2000-1-4" 的值
        expected = s2["2000-1-4"]
        # 获取 s2 中 Timestamp 为 "2000-1-4" 的实际结果
        result = s2[Timestamp("2000-1-4")]
        # 使用测试模块的 assert_series_equal 方法比较 result 和 expected
        tm.assert_series_equal(result, expected)

        # 获取 ser 中 Timestamp 为 "2000-1-4" 的预期结果
        result = ser[Timestamp("2000-1-4")]
        # 预期结果为 ser 中索引为 "2000-1-4" 的值
        expected = ser["2000-1-4"]
        # 使用测试模块的 assert_series_equal 方法比较 result 和 expected
        tm.assert_series_equal(result, expected)

        # 创建 DataFrame df2，将 ser 作为其数据
        df2 = DataFrame(ser)
        # 预期结果为 df2 中索引为 "2000-1-4" 的结果
        expected = df2.xs("2000-1-4")
        # 获取 df2 中 Timestamp 为 "2000-1-4" 的实际结果
        result = df2.loc[Timestamp("2000-1-4")]
        # 使用测试模块的 assert_frame_equal 方法比较 result 和 expected
        tm.assert_frame_equal(result, expected)

    def test_partial_slice_requires_monotonicity(self):
        # 自 2.0 版本起禁止（GH 37819）
        ser = Series(np.arange(10), date_range("2014-01-01", periods=10))

        # 选择 ser 的非单调部分
        nonmonotonic = ser.iloc[[3, 5, 4]]
        timestamp = Timestamp("2014-01-10")
        # 使用 pytest 模块的 raises 方法，预期 KeyError 匹配给定的错误信息
        with pytest.raises(
            KeyError, match="Value based partial slicing on non-monotonic"
        ):
            nonmonotonic["2014-01-10":]

        # 使用 pytest 模块的 raises 方法，预期 KeyError 匹配给定的错误信息
        with pytest.raises(KeyError, match=r"Timestamp\('2014-01-10 00:00:00'\)"):
            nonmonotonic[timestamp:]

        # 使用 pytest 模块的 raises 方法，预期 KeyError 匹配给定的错误信息
        with pytest.raises(
            KeyError, match="Value based partial slicing on non-monotonic"
        ):
            nonmonotonic.loc["2014-01-10":]

        # 使用 pytest 模块的 raises 方法，预期 KeyError 匹配给定的错误信息
        with pytest.raises(KeyError, match=r"Timestamp\('2014-01-10 00:00:00'\)"):
            nonmonotonic.loc[timestamp:]
    def test_loc_datetime_length_one(self):
        # GH16071
        # 创建一个包含单列的 DataFrame，索引为特定日期范围
        df = DataFrame(
            columns=["1"],
            index=date_range("2016-10-01T00:00:00", "2016-10-01T23:59:59"),
        )
        # 使用 datetime 对象进行切片选择，并验证结果与原始 DataFrame 相等
        result = df.loc[datetime(2016, 10, 1) :]
        tm.assert_frame_equal(result, df)

        # 使用时间戳字符串进行切片选择，并验证结果与原始 DataFrame 相等
        result = df.loc["2016-10-01T00:00:00":]
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize(
        "start",
        [
            "2018-12-02 21:50:00+00:00",
            Timestamp("2018-12-02 21:50:00+00:00"),
            Timestamp("2018-12-02 21:50:00+00:00").to_pydatetime(),
        ],
    )
    @pytest.mark.parametrize(
        "end",
        [
            "2018-12-02 21:52:00+00:00",
            Timestamp("2018-12-02 21:52:00+00:00"),
            Timestamp("2018-12-02 21:52:00+00:00").to_pydatetime(),
        ],
    )
    def test_getitem_with_datestring_with_UTC_offset(self, start, end):
        # GH 24076
        # 创建一个包含特定时区偏移的日期范围索引
        idx = date_range(
            start="2018-12-02 14:50:00-07:00",
            end="2018-12-02 14:50:00-07:00",
            freq="1min",
        )
        # 创建一个单列 DataFrame，索引为特定日期范围
        df = DataFrame(1, index=idx, columns=["A"])
        # 使用时间范围字符串进行切片选择，并验证结果与预期的 DataFrame 切片相等
        result = df[start:end]
        expected = df.iloc[0:3, :]
        tm.assert_frame_equal(result, expected)

        # GH 16785
        # 将 start 和 end 转换为字符串形式
        start = str(start)
        end = str(end)
        # 使用 pytest 断言检查索引选择中的 ValueError 异常
        with pytest.raises(ValueError, match="Both dates must"):
            df[start : end[:-4] + "1:00"]

        # 将 DataFrame 的时区信息移除
        df = df.tz_localize(None)
        # 使用 pytest 断言检查索引选择中的 ValueError 异常
        with pytest.raises(ValueError, match="The index must be timezone"):
            df[start:end]

    def test_slice_reduce_to_series(self):
        # GH 27516
        # 创建一个包含特定日期范围的 DataFrame，使用月末频率
        df = DataFrame(
            {"A": range(24)}, index=date_range("2000", periods=24, freq="ME")
        )
        # 创建一个预期的 Series，使用相同的日期范围和频率，验证结果与预期的 Series 相等
        expected = Series(
            range(12), index=date_range("2000", periods=12, freq="ME"), name="A"
        )
        # 使用 loc 进行切片选择，并验证结果与预期的 Series 相等
        result = df.loc["2000", "A"]
        tm.assert_series_equal(result, expected)
```