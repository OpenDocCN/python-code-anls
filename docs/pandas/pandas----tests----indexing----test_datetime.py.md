# `D:\src\scipysrc\pandas\pandas\tests\indexing\test_datetime.py`

```
import re  # 导入正则表达式模块

import pytest  # 导入 pytest 测试框架

import pandas as pd  # 导入 pandas 库并使用 pd 别名
from pandas import (  # 从 pandas 中导入多个类和函数
    DataFrame,  # 数据框类
    Index,  # 索引类
    Series,  # 系列类
    Timestamp,  # 时间戳类
    date_range,  # 日期范围函数
)
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestDatetimeIndex:
    def test_get_loc_naive_dti_aware_str_deprecated(self):
        # GH#46903
        ts = Timestamp("20130101")._value  # 创建时间戳对象并获取其内部值
        dti = pd.DatetimeIndex([ts + 50 + i for i in range(100)])  # 创建日期时间索引对象
        ser = Series(range(100), index=dti)  # 创建系列对象，指定索引为日期时间索引

        key = "2013-01-01 00:00:00.000000050+0000"  # 设置待查找的键值
        msg = re.escape(repr(key))  # 生成用于匹配异常消息的正则表达式字符串
        with pytest.raises(KeyError, match=msg):  # 检查是否抛出预期的 KeyError 异常
            ser[key]

        with pytest.raises(KeyError, match=msg):  # 再次检查是否抛出预期的 KeyError 异常
            dti.get_loc(key)

    def test_indexing_with_datetime_tz(self):
        # GH#8260
        # 支持带时区的 datetime64

        idx = Index(date_range("20130101", periods=3, tz="US/Eastern"), name="foo")  # 创建带时区的索引对象
        dr = date_range("20130110", periods=3)  # 创建日期范围对象
        df = DataFrame({"A": idx, "B": dr})  # 创建数据框，包含带时区的索引和日期范围
        df["C"] = idx  # 添加新列 C，并使用带时区的索引对象作为数据
        df.iloc[1, 1] = pd.NaT  # 将指定位置的值设置为 NaT（Not a Time）
        df.iloc[1, 2] = pd.NaT  # 将指定位置的值设置为 NaT

        expected = Series(
            [Timestamp("2013-01-02 00:00:00-0500", tz="US/Eastern"), pd.NaT, pd.NaT],  # 创建预期的系列对象
            index=list("ABC"),  # 设置索引标签
            dtype="object",  # 指定数据类型为对象
            name=1,  # 设置系列的名称
        )

        # 索引操作
        result = df.iloc[1]  # 使用整数位置索引获取行数据
        tm.assert_series_equal(result, expected)  # 断言获取的系列结果与预期相等
        result = df.loc[1]  # 使用标签索引获取行数据
        tm.assert_series_equal(result, expected)  # 断言获取的系列结果与预期相等

    def test_indexing_fast_xs(self):
        # 索引 - fast_xs
        df = DataFrame({"a": date_range("2014-01-01", periods=10, tz="UTC")})  # 创建数据框，包含带时区的日期范围
        result = df.iloc[5]  # 使用整数位置索引获取行数据
        expected = Series(
            [Timestamp("2014-01-06 00:00:00+0000", tz="UTC")],  # 创建预期的系列对象
            index=["a"],  # 设置索引标签
            name=5,  # 设置系列的名称
            dtype="M8[ns, UTC]",  # 指定数据类型为 datetime64[ns, UTC]
        )
        tm.assert_series_equal(result, expected)  # 断言获取的系列结果与预期相等

        result = df.loc[5]  # 使用标签索引获取行数据
        tm.assert_series_equal(result, expected)  # 断言获取的系列结果与预期相等

        # 索引 - boolean
        result = df[df.a > df.a[3]]  # 使用布尔索引获取满足条件的行数据
        expected = df.iloc[4:]  # 设置预期的数据框，包含符合条件的行数据
        tm.assert_frame_equal(result, expected)  # 断言获取的数据框结果与预期相等

    def test_consistency_with_tz_aware_scalar(self):
        # xef gh-12938
        # 使用相同的带时区标量进行多种索引方式
        df = Series([Timestamp("2016-03-30 14:35:25", tz="Europe/Brussels")]).to_frame()  # 创建系列，并转换为数据框

        df = pd.concat([df, df]).reset_index(drop=True)  # 拼接数据框，并重置索引
        expected = Timestamp("2016-03-30 14:35:25+0200", tz="Europe/Brussels")  # 创建预期的带时区时间戳对象

        result = df[0][0]  # 使用链式索引获取元素
        assert result == expected  # 断言获取的结果与预期相等

        result = df.iloc[0, 0]  # 使用整数位置索引获取元素
        assert result == expected  # 断言获取的结果与预期相等

        result = df.loc[0, 0]  # 使用标签索引获取元素
        assert result == expected  # 断言获取的结果与预期相等

        result = df.iat[0, 0]  # 使用整数位置快速获取元素
        assert result == expected  # 断言获取的结果与预期相等

        result = df.at[0, 0]  # 使用标签快速获取元素
        assert result == expected  # 断言获取的结果与预期相等

        result = df[0].loc[0]  # 先使用链式索引获取列，再使用标签索引获取元素
        assert result == expected  # 断言获取的结果与预期相等

        result = df[0].at[0]  # 先使用链式索引获取列，再使用标签快速获取元素
        assert result == expected  # 断言获取的结果与预期相等
    # 定义一个测试方法，用于测试带有 DateTimeIndex 和时区的索引操作
    def test_indexing_with_datetimeindex_tz(self, indexer_sl):
        # GH 12050
        # 在带有时区的 DateTimeIndex 上进行索引操作
        index = date_range("2015-01-01", periods=2, tz="utc")

        # 创建一个 Series 对象，包含两个整数，使用带有时区的 DateTimeIndex 作为索引
        ser = Series(range(2), index=index, dtype="int64")

        # 使用列表式索引进行迭代
        for sel in (index, list(index)):
            # 对 Series 对象进行索引操作并获取结果
            result = indexer_sl(ser)[sel]
            # 复制期望的 Series 对象
            expected = ser.copy()
            # 如果 sel 不是 index，调整期望的索引使其频率为 None
            if sel is not index:
                expected.index = expected.index._with_freq(None)
            # 断言两个 Series 对象是否相等
            tm.assert_series_equal(result, expected)

            # 对 Series 对象进行设置索引值的操作
            result = ser.copy()
            indexer_sl(result)[sel] = 1
            # 创建一个新的 Series 对象，所有值为 1，使用带有时区的 DateTimeIndex 作为索引
            expected = Series(1, index=index)
            # 断言两个 Series 对象是否相等
            tm.assert_series_equal(result, expected)

        # 单个元素的索引操作

        # 获取单个元素的值
        assert indexer_sl(ser)[index[1]] == 1

        # 对 Series 对象进行设置单个元素索引值的操作
        result = ser.copy()
        indexer_sl(result)[index[1]] = 5
        # 创建一个新的 Series 对象，第二个元素值为 5，使用带有时区的 DateTimeIndex 作为索引
        expected = Series([0, 5], index=index)
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，用于测试带有纳秒精度和时区的索引操作
    def test_nanosecond_getitem_setitem_with_tz(self):
        # GH 11679
        # 创建一个包含日期时间字符串的列表
        data = ["2016-06-28 08:30:00.123456789"]
        # 创建一个 DatetimeIndex 对象，使用纳秒和美国/芝加哥时区
        index = pd.DatetimeIndex(data, dtype="datetime64[ns, America/Chicago]")
        # 创建一个 DataFrame 对象，包含一个整数列 'a'，使用 DatetimeIndex 作为索引
        df = DataFrame({"a": [10]}, index=index)

        # 获取 DataFrame 对象的一个特定索引处的 Series 对象
        result = df.loc[df.index[0]]
        # 创建一个预期的 Series 对象，值为 10，索引为 df.index[0]
        expected = Series(10, index=["a"], name=df.index[0])
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

        # 对 DataFrame 对象进行设置特定索引处的值的操作
        result = df.copy()
        result.loc[df.index[0], "a"] = -1
        # 创建一个新的 DataFrame 对象，包含整数列 'a'，值为 -1，使用 DatetimeIndex 作为索引
        expected = DataFrame(-1, index=index, columns=["a"])
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试使用毫秒分辨率的字符串切片索引操作
    def test_getitem_str_slice_millisecond_resolution(self, frame_or_series):
        # GH#33589

        # 创建一个列表，包含时间戳字符串作为索引
        keys = [
            "2017-10-25T16:25:04.151",
            "2017-10-25T16:25:04.252",
            "2017-10-25T16:50:05.237",
            "2017-10-25T16:50:05.238",
        ]
        # 使用给定的索引和数值创建一个 Series 或者 DataFrame 对象
        obj = frame_or_series(
            [1, 2, 3, 4],
            index=[Timestamp(x) for x in keys],
        )
        # 获取 obj 对象中指定切片范围的结果
        result = obj[keys[1] : keys[2]]
        # 创建一个预期的 Series 或者 DataFrame 对象，包含指定切片范围的值和索引
        expected = frame_or_series(
            [2, 3],
            index=[
                Timestamp(keys[1]),
                Timestamp(keys[2]),
            ],
        )
        # 断言两个对象是否相等
        tm.assert_equal(result, expected)

    # 定义一个测试方法，用于测试使用 PyArrow 索引的索引操作
    def test_getitem_pyarrow_index(self, frame_or_series):
        # GH 53644
        # 导入 pytest 的 pyarrow 插件，如果未安装则跳过该测试
        pytest.importorskip("pyarrow")
        # 使用指定的索引和数值范围创建一个 Series 或者 DataFrame 对象
        obj = frame_or_series(
            range(5),
            index=date_range("2020", freq="D", periods=5).astype(
                "timestamp[us][pyarrow]"
            ),
        )
        # 获取 obj 对象中除了最后三个索引之外的结果
        result = obj.loc[obj.index[:-3]]
        # 创建一个预期的 Series 或者 DataFrame 对象，包含除了最后三个索引之外的值和索引
        expected = frame_or_series(
            range(2),
            index=date_range("2020", freq="D", periods=2).astype(
                "timestamp[us][pyarrow]"
            ),
        )
        # 断言两个对象是否相等
        tm.assert_equal(result, expected)
```