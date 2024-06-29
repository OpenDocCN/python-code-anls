# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_combine_first.py`

```
# 导入 datetime 模块中的 datetime 类
from datetime import datetime

# 导入 numpy 库，并使用别名 np
import numpy as np

# 导入 pandas 库，并从中导入 Period、Series、date_range、period_range、to_datetime 函数
import pandas as pd
from pandas import (
    Period,
    Series,
    date_range,
    period_range,
    to_datetime,
)

# 导入 pandas 内部测试模块，使用别名 tm
import pandas._testing as tm

# 定义一个测试类 TestCombineFirst
class TestCombineFirst:

    # 定义测试方法 test_combine_first_period_datetime
    def test_combine_first_period_datetime(self):
        # 创建日期范围对象 didx，包含从 "1950-01-31" 到 "1950-07-31" 的月末日期
        didx = date_range(start="1950-01-31", end="1950-07-31", freq="ME")
        
        # 创建周期范围对象 pidx，包含从 Period("1950-1") 到 Period("1950-7") 的每月周期
        pidx = period_range(start=Period("1950-1"), end=Period("1950-7"), freq="M")
        
        # 遍历 didx 和 pidx 两个日期范围对象
        for idx in [didx, pidx]:
            # 创建 Series 对象 a，包含指定索引 idx 的数据，其中包括 NaN 值
            a = Series([1, np.nan, np.nan, 4, 5, np.nan, 7], index=idx)
            
            # 创建 Series 对象 b，包含指定索引 idx 的数据，所有值为 9
            b = Series([9, 9, 9, 9, 9, 9, 9], index=idx)

            # 使用 a 的数据与 b 的数据进行合并，优先使用 a 中的数据
            result = a.combine_first(b)
            
            # 创建期望的 Series 对象 expected，包含预期的数据和索引
            expected = Series([1, 9, 9, 4, 5, 9, 7], index=idx, dtype=np.float64)
            
            # 断言 result 和 expected 的内容相等
            tm.assert_series_equal(result, expected)

    # 定义测试方法 test_combine_first_name，参数为 datetime_series
    def test_combine_first_name(self, datetime_series):
        # 使用 datetime_series 和其前 5 个元素进行合并
        result = datetime_series.combine_first(datetime_series[:5])
        
        # 断言 result 的名称与 datetime_series 的名称相同
        assert result.name == datetime_series.name

    # 定义测试方法 test_combine_first
    def test_combine_first(self):
        # 创建一个包含 0 到 19 的浮点数序列 values
        values = np.arange(20, dtype=np.float64)
        
        # 创建 Series 对象 series，使用 values 作为数据，0 到 19 作为索引
        series = Series(values, index=np.arange(20, dtype=np.int64))

        # 创建 series 的副本 series_copy，并将其每隔一个元素设为 NaN
        series_copy = series * 2
        series_copy[::2] = np.nan

        # 对比 series 和 series_copy，优先使用 series_copy 中的数据进行合并
        combined = series.combine_first(series_copy)
        
        # 断言合并后的结果与原始的 series 相等
        tm.assert_series_equal(combined, series)

        # 使用 series 中的数据填充 series_copy 中的空缺值
        combined = series_copy.combine_first(series)
        
        # 断言合并后的结果所有值都是有限的
        assert np.isfinite(combined).all()

        # 断言合并后的结果中，每隔一个元素的部分与 series 相等
        tm.assert_series_equal(combined[::2], series[::2])
        
        # 断言合并后的结果中，除了每隔一个元素之外的部分与 series_copy 相等
        tm.assert_series_equal(combined[1::2], series_copy[1::2])

        # 创建一个索引为字符串的 Index 对象 index
        index = pd.Index([str(i) for i in range(20)])
        
        # 创建包含标准正态分布随机数的浮点数序列 floats，并指定索引 index
        floats = Series(np.random.default_rng(2).standard_normal(20), index=index)
        
        # 创建包含字符串的对象序列 strings，指定每隔一个元素的索引 index
        strings = Series([str(i) for i in range(10)], index=index[::2], dtype=object)

        # 使用 strings 和 floats 进行合并
        combined = strings.combine_first(floats)

        # 断言合并后的结果与 strings 中每隔一个元素的部分相等
        tm.assert_series_equal(strings, combined.loc[index[::2]])
        
        # 断言合并后的结果与 floats 中每隔一个元素的部分（转换为对象类型）相等
        tm.assert_series_equal(floats[1::2].astype(object), combined.loc[index[1::2]])

        # 创建一个包含整数和浮点数的 Series 对象 ser
        ser = Series([1.0, 2, 3], index=[0, 1, 2])
        
        # 创建一个空的 Series 对象 empty，指定 dtype 为 object
        empty = Series([], index=[], dtype=object)
        
        # 使用 empty 和 ser 进行合并
        result = ser.combine_first(empty)
        
        # 将 ser 的索引类型转换为对象类型
        ser.index = ser.index.astype("O")
        
        # 断言合并后的结果与 ser 转换为对象类型后的内容相等
        tm.assert_series_equal(result, ser.astype(object))

    # 定义测试方法 test_combine_first_dt64，参数为 unit
    def test_combine_first_dt64(self, unit):
        # 创建包含日期字符串的 Series 对象 s0，转换为指定单位的 datetime64 类型
        s0 = to_datetime(Series(["2010", np.nan])).dt.as_unit(unit)
        
        # 创建包含日期字符串的 Series 对象 s1
        s1 = to_datetime(Series([np.nan, "2011"])).dt.as_unit(unit)
        
        # 使用 s0 和 s1 进行合并
        rs = s0.combine_first(s1)
        
        # 创建期望的日期字符串的 Series 对象 xp，转换为指定单位的 datetime64 类型
        xp = to_datetime(Series(["2010", "2011"])).dt.as_unit(unit)
        
        # 断言合并后的结果与期望的结果相等
        tm.assert_series_equal(rs, xp)

        # 创建包含日期字符串的 Series 对象 s0，转换为指定单位的 datetime64 类型
        s0 = to_datetime(Series(["2010", np.nan])).dt.as_unit(unit)
        
        # 创建包含日期字符串的 Series 对象 s1
        s1 = Series([np.nan, "2011"])
        
        # 使用 s0 和 s1 进行合并
        rs = s0.combine_first(s1)

        # 创建期望的 Series 对象 xp，包含日期和对象类型
        xp = Series([datetime(2010, 1, 1), "2011"], dtype=f"datetime64[{unit}]")
        
        # 断言合并后的结果与期望的结果相等
        tm.assert_series_equal(rs, xp)
    def test_combine_first_dt_tz_values(self, tz_naive_fixture):
        # 测试函数：测试在具有时区信息的日期时间索引中，combine_first 方法的行为
        ser1 = Series(
            pd.DatetimeIndex(["20150101", "20150102", "20150103"], tz=tz_naive_fixture),
            name="ser1",
        )
        ser2 = Series(
            pd.DatetimeIndex(["20160514", "20160515", "20160516"], tz=tz_naive_fixture),
            index=[2, 3, 4],
            name="ser2",
        )
        # 执行 combine_first 操作，将 ser2 中的值合并到 ser1 中
        result = ser1.combine_first(ser2)
        # 期望的结果，包含了合并后的日期时间索引和时区信息
        exp_vals = pd.DatetimeIndex(
            ["20150101", "20150102", "20150103", "20160515", "20160516"],
            tz=tz_naive_fixture,
        )
        exp = Series(exp_vals, name="ser1")
        # 使用测试工具比较实际结果和期望结果
        tm.assert_series_equal(exp, result)

    def test_combine_first_timezone_series_with_empty_series(self):
        # 测试函数：测试在一个时间序列与一个空时间序列进行 combine_first 操作的行为
        # GH 41800
        # 创建一个从 2021 年 1 月 1 日 1 点到 10 点的时间索引，每小时频率，欧洲/罗马时区
        time_index = date_range(
            datetime(2021, 1, 1, 1),
            datetime(2021, 1, 1, 10),
            freq="h",
            tz="Europe/Rome",
        )
        # 创建两个时间序列，s1 包含完整的索引和数据，s2 索引与 s1 相同但数据为空
        s1 = Series(range(10), index=time_index)
        s2 = Series(index=time_index)
        # 执行 combine_first 操作，将 s2 中的值合并到 s1 中
        result = s1.combine_first(s2)
        # 期望的结果是将 s1 中的数据类型转换为 np.float64
        tm.assert_series_equal(result, s1.astype(np.float64))

    def test_combine_first_preserves_dtype(self):
        # 测试函数：测试 combine_first 方法在保留数据类型方面的行为
        # GH51764
        # 创建两个时间序列，s1 包含整数数据，s2 包含不同的数据
        s1 = Series([1666880195890293744, 1666880195890293837])
        s2 = Series([1, 2, 3])
        # 执行 combine_first 操作
        result = s1.combine_first(s2)
        # 期望的结果是保留 s1 中的数据类型
        expected = Series([1666880195890293744, 1666880195890293837, 3])
        tm.assert_series_equal(result, expected)

    def test_combine_mixed_timezone(self):
        # 测试函数：测试在混合时区时间序列上使用 combine_first 方法的行为
        # GH 26283
        # 创建一个拥有统一时区的时间序列和一个包含不同时区的时间序列
        uniform_tz = Series({pd.Timestamp("2019-05-01", tz="UTC"): 1.0})
        multi_tz = Series(
            {
                pd.Timestamp("2019-05-01 01:00:00+0100", tz="Europe/London"): 2.0,
                pd.Timestamp("2019-05-02", tz="UTC"): 3.0,
            }
        )
        # 执行 combine_first 操作，将 multi_tz 中的数据合并到 uniform_tz 中
        result = uniform_tz.combine_first(multi_tz)
        # 期望的结果是一个包含合并后数据的时间序列，使用统一的 UTC 时区
        expected = Series(
            [1.0, 3.0],
            index=pd.Index(
                [
                    pd.Timestamp("2019-05-01 00:00:00+00:00", tz="UTC"),
                    pd.Timestamp("2019-05-02 00:00:00+00:00", tz="UTC"),
                ],
                dtype="object",
            ),
        )
        # 使用测试工具比较实际结果和期望结果
        tm.assert_series_equal(result, expected)
```