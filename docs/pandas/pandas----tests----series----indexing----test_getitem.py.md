# `D:\src\scipysrc\pandas\pandas\tests\series\indexing\test_getitem.py`

```
"""
Series.__getitem__ test classes are organized by the type of key passed.
"""

# 从 datetime 模块中导入需要的类
from datetime import (
    date,
    datetime,
    time,
)

# 导入 numpy 和 pytest 库
import numpy as np
import pytest

# 从 pandas._libs.tslibs 中导入 conversion 和 timezones 模块
from pandas._libs.tslibs import (
    conversion,
    timezones,
)

# 从 pandas.compat.numpy 中导入 np_version_gt2 函数
from pandas.compat.numpy import np_version_gt2

# 从 pandas.core.dtypes.common 中导入 is_scalar 函数
from pandas.core.dtypes.common import is_scalar

# 导入 pandas 库并重命名为 pd
import pandas as pd
# 从 pandas 中导入需要的类和函数
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    Timestamp,
    date_range,
    period_range,
    timedelta_range,
)
# 导入 pandas 的测试工具模块并重命名为 tm
import pandas._testing as tm

# 从 pandas.core.indexing 中导入 IndexingError 异常类
from pandas.core.indexing import IndexingError

# 从 pandas.tseries.offsets 中导入 BDay 类
from pandas.tseries.offsets import BDay


# 定义测试类 TestSeriesGetitemScalars
class TestSeriesGetitemScalars:
    
    # 测试方法：测试使用对象索引为浮点数和字符串的情况
    def test_getitem_object_index_float_string(self):
        # 创建一个 Series，索引包含字符串和浮点数
        ser = Series([1] * 4, index=Index(["a", "b", "c", 1.0]))
        # 断言索引为 "a" 的值为 1
        assert ser["a"] == 1
        # 断言索引为 1.0 的值为 1
        assert ser[1.0] == 1

    # 测试方法：测试使用浮点数作为索引键，元组作为值的情况
    def test_getitem_float_keys_tuple_values(self):
        # see GH#13509

        # 创建一个 Series，其索引为唯一的浮点数
        ser = Series([(1, 1), (2, 2), (3, 3)], index=[0.0, 0.1, 0.2], name="foo")
        # 获取索引为 0.0 的值
        result = ser[0.0]
        # 断言结果为 (1, 1)
        assert result == (1, 1)

        # 创建一个 Series，其索引中有重复的浮点数
        expected = Series([(1, 1), (2, 2)], index=[0.0, 0.0], name="foo")
        ser = Series([(1, 1), (2, 2), (3, 3)], index=[0.0, 0.0, 0.2], name="foo")

        # 获取索引为 0.0 的值
        result = ser[0.0]
        # 使用测试工具函数验证结果与预期是否相等
        tm.assert_series_equal(result, expected)

    # 测试方法：测试使用未识别标量的情况
    def test_getitem_unrecognized_scalar(self):
        # see GH#32684 a scalar key that is not recognized by lib.is_scalar

        # 创建一个 Series，索引包含两种类型的数据类型对象
        ser = Series([1, 2], index=[np.dtype("O"), np.dtype("i8")])

        # 获取索引为 1 的键
        key = ser.index[1]

        # 使用索引键获取 Series 的值
        result = ser[key]
        # 断言结果为 2
        assert result == 2

    # 测试方法：测试使用超出边界索引的情况，应该抛出 KeyError 异常
    def test_getitem_negative_out_of_bounds(self):
        # 创建一个长度为 10 的 Series，所有值为 "a"，索引也为 "a"
        ser = Series(["a"] * 10, index=["a"] * 10)

        # 使用 pytest 的断言捕获 KeyError 异常，确保索引为 -11 时抛出异常
        with pytest.raises(KeyError, match="^-11$"):
            ser[-11]

    # 测试方法：测试使用超出范围的索引时，应该抛出 KeyError 异常
    def test_getitem_out_of_bounds_indexerror(self, datetime_series):
        # don't segfault, GH#495
        # 获取 datetime_series 的长度
        N = len(datetime_series)
        # 使用 pytest 的断言捕获 KeyError 异常，确保索引为 N 时抛出异常
        with pytest.raises(KeyError, match=str(N)):
            datetime_series[N]

    # 测试方法：测试使用空的 RangeIndex 时，使用 int 索引应该抛出 KeyError 异常
    def test_getitem_out_of_bounds_empty_rangeindex_keyerror(self):
        # GH#917
        # 创建一个空的 Series，数据类型为 object
        ser = Series([], dtype=object)
        # 使用 pytest 的断言捕获 KeyError 异常，确保索引为 -1 时抛出异常
        with pytest.raises(KeyError, match="-1"):
            ser[-1]
    # 测试使用整数索引时抛出 KeyError 异常
    def test_getitem_keyerror_with_integer_index(self, any_int_numpy_dtype):
        # 设置 dtype 为任意整数类型
        dtype = any_int_numpy_dtype
        # 创建 Series 对象，使用标准正态分布的随机数填充数据，设置索引为 [0, 0, 1, 1, 2, 2]
        ser = Series(
            np.random.default_rng(2).standard_normal(6),
            index=Index([0, 0, 1, 1, 2, 2], dtype=dtype),
        )

        # 断言索引为 5 的元素会抛出 KeyError 异常
        with pytest.raises(KeyError, match=r"^5$"):
            ser[5]

        # 断言索引为 'c' 的元素会抛出 KeyError 异常
        with pytest.raises(KeyError, match=r"^'c'$"):
            ser["c"]

        # 创建一个非单调递增的 Series 对象，使用标准正态分布的随机数填充数据，设置索引为 [2, 2, 0, 0, 1, 1]
        # 非单调递增意味着索引顺序并不是严格递增的
        ser = Series(
            np.random.default_rng(2).standard_normal(6), index=[2, 2, 0, 0, 1, 1]
        )

        # 断言索引为 5 的元素会抛出 KeyError 异常
        with pytest.raises(KeyError, match=r"^5$"):
            ser[5]

        # 断言索引为 'c' 的元素会抛出 KeyError 异常
        with pytest.raises(KeyError, match=r"^'c'$"):
            ser["c"]

    # 测试使用 np.int64 类型索引时抛出 KeyError 异常
    def test_getitem_int64(self, datetime_series):
        # 如果 numpy 版本大于 2，则匹配包含 np.int64(5) 的 KeyError 消息
        if np_version_gt2:
            msg = r"^np.int64\(5\)$"
        else:
            msg = "^5$"
        # 创建 np.int64 类型的索引 idx
        idx = np.int64(5)
        # 断言索引为 idx 的元素会抛出匹配 msg 的 KeyError 异常
        with pytest.raises(KeyError, match=msg):
            datetime_series[idx]

    # 测试获取 Series 对象全部范围的元素
    def test_getitem_full_range(self):
        # 使用 range(5) 创建 Series 对象，索引和数据都是 [0, 1, 2, 3, 4]
        ser = Series(range(5), index=list(range(5)))
        # 使用索引为 [0, 1, 2, 3, 4] 的列表获取结果
        result = ser[list(range(5))]
        # 断言结果与原始 Series 相等
        tm.assert_series_equal(result, ser)

    # ------------------------------------------------------------------
    # 使用 DatetimeIndex 的 Series

    # 使用带时区信息的 pydatetime 进行索引的测试
    @pytest.mark.parametrize("tzstr", ["Europe/Berlin", "dateutil/Europe/Berlin"])
    def test_getitem_pydatetime_tz(self, tzstr):
        # 获取时区 tz 对象
        tz = timezones.maybe_get_tz(tzstr)

        # 创建一个从 "2012-12-24 16:00" 到 "2012-12-24 18:00"，每小时一个频率的 DatetimeIndex
        index = date_range(
            start="2012-12-24 16:00", end="2012-12-24 18:00", freq="h", tz=tzstr
        )
        # 创建 Series 对象，索引为 index，数据为 index 的小时部分
        ts = Series(index=index, data=index.hour)
        # 创建一个带有时区信息的 Timestamp 对象
        time_pandas = Timestamp("2012-12-24 17:00", tz=tzstr)

        # 创建一个 datetime 对象，并使用给定的时区 tz 进行本地化
        dt = datetime(2012, 12, 24, 17, 0)
        time_datetime = conversion.localize_pydatetime(dt, tz)
        # 断言使用 time_pandas 和 time_datetime 进行索引的结果相等
        assert ts[time_pandas] == ts[time_datetime]

    # 使用字符串索引进行时区感知测试
    @pytest.mark.parametrize("tz", ["US/Eastern", "dateutil/US/Eastern"])
    def test_string_index_alias_tz_aware(self, tz):
        # 创建一个带有时区信息的日期范围 rng
        rng = date_range("1/1/2000", periods=10, tz=tz)
        # 创建 Series 对象，使用标准正态分布的随机数填充数据，索引为 rng
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

        # 从 Series 中获取日期为 "1/3/2000" 的结果
        result = ser["1/3/2000"]
        # 使用 iloc 方法获取索引为 2 的元素，并进行近似断言
        tm.assert_almost_equal(result, ser.iloc[2])

    # 使用时间对象进行索引的测试
    def test_getitem_time_object(self):
        # 创建一个时间范围 rng，频率为每 5 分钟
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        # 创建 Series 对象，使用标准正态分布的随机数填充数据，索引为 rng
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

        # 创建一个布尔掩码 mask，用于获取时间为 9:30 的数据
        mask = (rng.hour == 9) & (rng.minute == 30)
        # 使用时间对象 time(9, 30) 进行索引获取结果
        result = ts[time(9, 30)]
        # 从 ts 中使用 mask 获取期望的结果
        expected = ts[mask]
        # 将 result 的索引频率设置为 None 后，进行 Series 相等性断言
        result.index = result.index._with_freq(None)
        tm.assert_series_equal(result, expected)

    # ------------------------------------------------------------------
    # 使用 CategoricalIndex 的 Series
    def test_getitem_scalar_categorical_index(self):
        # 创建一个包含日期时间戳的分类数据
        cats = Categorical([Timestamp("12-31-1999"), Timestamp("12-31-2000")])

        # 使用分类数据作为索引创建一个 Series 对象
        ser = Series([1, 2], index=cats)

        # 预期结果是选择第一个元素
        expected = ser.iloc[0]
        # 使用分类索引值访问 Series 对象
        result = ser[cats[0]]
        # 断言结果与预期值相等
        assert result == expected

    def test_getitem_numeric_categorical_listlike_matches_scalar(self):
        # GH#15470

        # 创建一个 Series 对象，其索引为分类索引
        ser = Series(["a", "b", "c"], index=pd.CategoricalIndex([2, 1, 0]))

        # 0 被视为标签
        assert ser[0] == "c"

        # listlike 形式也应被视为标签
        res = ser[[0]]
        # 预期结果是最后一个元素的 Series 对象
        expected = ser.iloc[-1:]
        # 使用工具函数验证 Series 相等性
        tm.assert_series_equal(res, expected)

        res2 = ser[[0, 1, 2]]
        # 验证结果与逆序的 Series 相等
        tm.assert_series_equal(res2, ser.iloc[::-1])

    def test_getitem_integer_categorical_not_positional(self):
        # GH#14865

        # 创建一个 Series 对象，其索引为分类索引
        ser = Series(["a", "b", "c"], index=Index([1, 2, 3], dtype="category"))
        # 使用索引值获取值并断言结果
        assert ser.get(3) == "c"
        assert ser[3] == "c"

    def test_getitem_str_with_timedeltaindex(self):
        # 使用 timedelta_range 创建一个时间差范围
        rng = timedelta_range("1 day 10:11:12", freq="h", periods=500)
        # 创建一个 Series 对象，其索引为时间差范围
        ser = Series(np.arange(len(rng)), index=rng)

        key = "6 days, 23:11:12"
        # 获取键值在索引中的位置
        indexer = rng.get_loc(key)
        assert indexer == 133

        # 使用时间差索引获取 Series 对象中的值
        result = ser[key]
        assert result == ser.iloc[133]

        msg = r"^Timedelta\('50 days 00:00:00'\)$"
        # 使用 pytest 验证抛出 KeyError 异常
        with pytest.raises(KeyError, match=msg):
            rng.get_loc("50 days")
        with pytest.raises(KeyError, match=msg):
            ser["50 days"]

    def test_getitem_bool_index_positional(self):
        # GH#48653

        # 创建一个包含布尔索引的 Series 对象
        ser = Series({True: 1, False: 0})
        # 使用布尔索引访问 Series 对象，验证抛出 KeyError 异常
        with pytest.raises(KeyError, match="^0$"):
            ser[0]
    # 定义一个测试类 TestSeriesGetitemSlices，用于测试 Series 对象的切片操作
    class TestSeriesGetitemSlices:
        
        # 测试从 DatetimeIndex 中获取部分字符串切片
        def test_getitem_partial_str_slice_with_datetimeindex(self):
            # GH#34860
            # 创建一个日期范围数组 arr，从 "1/1/2008" 到 "1/1/2009"
            arr = date_range("1/1/2008", "1/1/2009")
            # 将日期范围数组 arr 转换为 Series 对象
            ser = arr.to_series()
            # 对 Series 进行索引，获取年份为 "2008" 的数据
            result = ser["2008"]
    
            # 创建一个从 "2008-01-01" 到 "2008-12-31" 的日期范围 rng
            rng = date_range(start="2008-01-01", end="2008-12-31")
            # 创建一个期望的 Series 对象，使用日期范围 rng，指定索引为 rng
            expected = Series(rng, index=rng)
    
            # 断言 result 和 expected 的 Series 对象是否相等
            tm.assert_series_equal(result, expected)
    
        # 测试从 DatetimeIndex 中获取字符串切片
        def test_getitem_slice_strings_with_datetimeindex(self):
            # 创建一个 DatetimeIndex，包含日期字符串
            idx = DatetimeIndex(
                ["1/1/2000", "1/2/2000", "1/2/2000", "1/3/2000", "1/4/2000"]
            )
    
            # 创建一个 Series 对象，使用随机生成的标准正态分布数据，指定索引为 idx
            ts = Series(np.random.default_rng(2).standard_normal(len(idx)), index=idx)
    
            # 对 Series 进行切片，从 "1/2/2000" 开始到末尾
            result = ts["1/2/2000":]
            # 创建一个期望的 Series 对象，从索引 1 开始到末尾
            expected = ts[1:]
            # 断言 result 和 expected 的 Series 对象是否相等
            tm.assert_series_equal(result, expected)
    
            # 对 Series 进行切片，从 "1/2/2000" 到 "1/3/2000"
            result = ts["1/2/2000":"1/3/2000"]
            # 创建一个期望的 Series 对象，从索引 1 到 4
            expected = ts[1:4]
            # 断言 result 和 expected 的 Series 对象是否相等
            tm.assert_series_equal(result, expected)
    
        # 测试从 TimedeltaIndex 中获取部分字符串切片
        def test_getitem_partial_str_slice_with_timedeltaindex(self):
            # 创建一个时间间隔范围 rng，从 "1 day 10:11:12" 开始，频率为每小时，共 500 个周期
            rng = timedelta_range("1 day 10:11:12", freq="h", periods=500)
            # 创建一个 Series 对象，使用从 0 开始的数字作为数据，指定索引为 rng
            ser = Series(np.arange(len(rng)), index=rng)
    
            # 对 Series 进行切片，从 "5 day" 到 "6 day"
            result = ser["5 day":"6 day"]
            # 创建一个期望的 Series 对象，使用 iloc 方法从索引 86 到 134
            expected = ser.iloc[86:134]
            # 断言 result 和 expected 的 Series 对象是否相等
            tm.assert_series_equal(result, expected)
    
            # 对 Series 进行切片，从 "5 day" 到末尾
            result = ser["5 day":]
            # 创建一个期望的 Series 对象，使用 iloc 方法从索引 86 到末尾
            expected = ser.iloc[86:]
            # 断言 result 和 expected 的 Series 对象是否相等
            tm.assert_series_equal(result, expected)
    
            # 对 Series 进行切片，从开头到 "6 day"
            result = ser[:"6 day"]
            # 创建一个期望的 Series 对象，使用 iloc 方法从开头到索引 134
            expected = ser.iloc[:134]
            # 断言 result 和 expected 的 Series 对象是否相等
            tm.assert_series_equal(result, expected)
    
        # 测试从 TimedeltaIndex 中获取高分辨率部分字符串切片
        def test_getitem_partial_str_slice_high_reso_with_timedeltaindex(self):
            # 创建一个高分辨率的时间间隔范围 rng，从 "1 day 10:11:12" 开始，频率为每微秒，共 2000 个周期
            rng = timedelta_range("1 day 10:11:12", freq="us", periods=2000)
            # 创建一个 Series 对象，使用从 0 开始的数字作为数据，指定索引为 rng
            ser = Series(np.arange(len(rng)), index=rng)
    
            # 对 Series 进行切片，从 "1 day 10:11:12" 到末尾
            result = ser["1 day 10:11:12":]
            # 创建一个期望的 Series 对象，使用 iloc 方法从索引 0 到末尾
            expected = ser.iloc[0:]
            # 断言 result 和 expected 的 Series 对象是否相等
            tm.assert_series_equal(result, expected)
    
            # 对 Series 进行切片，从 "1 day 10:11:12.001" 到末尾
            result = ser["1 day 10:11:12.001":]
            # 创建一个期望的 Series 对象，使用 iloc 方法从索引 1000 到末尾
            expected = ser.iloc[1000:]
            # 断言 result 和 expected 的 Series 对象是否相等
            tm.assert_series_equal(result, expected)
    
            # 获取 Series 中特定的时间间隔，精确到 "1 days, 10:11:12.001001"
            result = ser["1 days, 10:11:12.001001"]
            # 断言 result 和 iloc 方法获取的 Series 对象是否相等
            assert result == ser.iloc[1001]
    
        # 测试二维切片操作，预期会引发 ValueError 异常，提示多维索引已弃用
        def test_getitem_slice_2d(self, datetime_series):
            # GH#30588 multi-dimensional indexing deprecated
            with pytest.raises(ValueError, match="Multi-dimensional indexing"):
                # 尝试对 datetime_series 进行多维切片操作
                datetime_series[:, np.newaxis]
    
        # 测试在特定情况下获取中位数切片引发的 bug
        def test_getitem_median_slice_bug(self):
            # 创建一个日期范围 index，从 "20090415" 到 "20090519"，频率为每 2 个工作日
            index = date_range("20090415", "20090519", freq="2B")
            # 创建一个 Series 对象，使用随机生成的标准正态分布数据，指定索引为 index
            ser = Series(np.random.default_rng(2).standard_normal(13), index=index)
    
            # 创建一个切片索引器，包含一个 slice 对象，从索引 6 到 7
            indexer = [slice(6, 7, None)]
            msg = "Indexing with a single-item list"
            with pytest.raises(ValueError, match=msg):
                # GH#31299
                # 尝试使用包含单个元素的列表进行索引，预期引发 ValueError 异常
                ser[indexer]
            # 对具有单个元素元组的情况进行处理
            result = ser[(indexer[0],)]
            # 创建一个期望的 Series 对象，使用 slice 对象从索引 6 到 7
            expected = ser[indexer[0]]
            # 断言 result 和 expected 的 Series 对象是否相等
            tm.assert_series_equal(result, expected)
    @pytest.mark.parametrize(
        "slc, positions",
        [
            [slice(date(2018, 1, 1), None), [0, 1, 2]],
            [slice(date(2019, 1, 2), None), [2]],
            [slice(date(2020, 1, 1), None), []],
            [slice(None, date(2020, 1, 1)), [0, 1, 2]],
            [slice(None, date(2019, 1, 1)), [0]],
        ],
    )
    # 定义测试函数，用于测试对 DatetimeIndex 进行切片操作的情况
    def test_getitem_slice_date(self, slc, positions):
        # 创建一个包含日期时间索引的 Series 对象
        ser = Series(
            [0, 1, 2],
            DatetimeIndex(["2019-01-01", "2019-01-01T06:00:00", "2019-01-02"]),
        )
        # 执行切片操作，获取结果
        result = ser[slc]
        # 根据 positions 列表构建预期结果
        expected = ser.take(positions)
        # 使用测试框架中的 assert 函数比较结果和预期结果是否相等
        tm.assert_series_equal(result, expected)

    # 定义测试函数，验证在对 DatetimeIndex 使用浮点数索引时是否会引发 TypeError
    def test_getitem_slice_float_raises(self, datetime_series):
        # 错误消息模板
        msg = (
            "cannot do slice indexing on DatetimeIndex with these indexers "
            r"\[{key}\] of type float"
        )
        # 断言调用时会引发 TypeError，并匹配特定的错误消息格式
        with pytest.raises(TypeError, match=msg.format(key=r"4\.0")):
            datetime_series[4.0:10.0]

        with pytest.raises(TypeError, match=msg.format(key=r"4\.5")):
            datetime_series[4.5:10.0]

    # 定义测试函数，测试 Series 对象在特定切片操作下的 bug 是否得到正确处理
    def test_getitem_slice_bug(self):
        # 创建一个整数索引的 Series 对象
        ser = Series(range(10), index=list(range(10)))
        # 测试负数索引切片时的结果是否正确
        result = ser[-12:]
        tm.assert_series_equal(result, ser)

        # 测试负数索引和正数索引组合切片时的结果是否正确
        result = ser[-7:]
        tm.assert_series_equal(result, ser[3:])

        # 测试负数索引切片时的结果是否正确
        result = ser[:-12]
        tm.assert_series_equal(result, ser[:0])

    # 定义测试函数，测试对整数索引进行切片操作的情况
    def test_getitem_slice_integers(self):
        # 创建一个随机数据的 Series 对象，使用整数索引
        ser = Series(
            np.random.default_rng(2).standard_normal(8),
            index=[2, 4, 6, 8, 10, 12, 14, 16],
        )

        # 测试从开头到索引为 4 的切片操作是否得到预期结果
        result = ser[:4]
        expected = Series(ser.values[:4], index=[2, 4, 6, 8])
        tm.assert_series_equal(result, expected)
class TestSeriesGetitemListLike:
    @pytest.mark.parametrize("box", [list, np.array, Index, Series])
    def test_getitem_no_matches(self, box):
        # GH#33462 we expect the same behavior for list/ndarray/Index/Series
        # 创建一个包含字符串 "A", "B" 的 Series 对象
        ser = Series(["A", "B"])

        # 创建包含字符串 "C" 的 Series 对象，并根据 box 类型转换为相应的类型对象
        key = Series(["C"], dtype=object)
        key = box(key)

        # 定义期望的错误消息模式
        msg = (
            r"None of \[Index\(\['C'\], dtype='object|string'\)\] are in the \[index\]"
        )
        # 断言通过 ser[key] 访问时会抛出 KeyError，并且错误消息符合预期的模式
        with pytest.raises(KeyError, match=msg):
            ser[key]

    def test_getitem_intlist_intindex_periodvalues(self):
        # 创建一个包含日期范围的 Series 对象
        ser = Series(period_range("2000-01-01", periods=10, freq="D"))

        # 使用整数列表作为索引来获取子集
        result = ser[[2, 4]]
        # 创建期望的 Series 对象
        exp = Series(
            [pd.Period("2000-01-03", freq="D"), pd.Period("2000-01-05", freq="D")],
            index=[2, 4],
            dtype="Period[D]",
        )
        # 断言两个 Series 对象相等
        tm.assert_series_equal(result, exp)
        # 断言结果的数据类型为 Period[D]
        assert result.dtype == "Period[D]"

    @pytest.mark.parametrize("box", [list, np.array, Index])
    def test_getitem_intlist_intervalindex_non_int(self, box):
        # 创建一个 IntervalIndex 对象
        dti = date_range("2000-01-03", periods=3)._with_freq(None)
        ii = pd.IntervalIndex.from_breaks(dti)
        ser = Series(range(len(ii)), index=ii)

        # 使用整数列表作为索引来获取子集，期望抛出 KeyError
        key = box([0])
        msg = r"None of \[Index\(\[0\], dtype='int(32|64)'\)\] are in the \[index\]"
        with pytest.raises(KeyError, match=msg):
            ser[key]

    @pytest.mark.parametrize("box", [list, np.array, Index])
    @pytest.mark.parametrize("dtype", [np.int64, np.float64, np.uint64])
    def test_getitem_intlist_multiindex_numeric_level(self, dtype, box):
        # 创建一个 MultiIndex 对象
        idx = Index(range(4)).astype(dtype)
        dti = date_range("2000-01-03", periods=3)
        mi = pd.MultiIndex.from_product([idx, dti])
        ser = Series(range(len(mi))[::-1], index=mi)

        # 使用整数列表作为索引来获取子集，期望抛出包含 "5" 的 KeyError
        key = box([5])
        with pytest.raises(KeyError, match="5"):
            ser[key]

    def test_getitem_uint_array_key(self, any_unsigned_int_numpy_dtype):
        # GH #37218
        # 创建一个整数类型的 Series 对象
        ser = Series([1, 2, 3])
        # 创建一个包含无符号整数的 numpy 数组作为索引
        key = np.array([4], dtype=any_unsigned_int_numpy_dtype)

        # 断言通过 ser[key] 访问时会抛出包含 "4" 的 KeyError
        with pytest.raises(KeyError, match="4"):
            ser[key]
        # 断言通过 ser.loc[key] 访问时会抛出包含 "4" 的 KeyError
        with pytest.raises(KeyError, match="4"):
            ser.loc[key]


class TestGetitemBooleanMask:
    def test_getitem_boolean(self, string_series):
        # 创建一个字符串类型的 Series 对象
        ser = string_series
        # 创建一个布尔掩码
        mask = ser > ser.median()

        # 使用布尔列表作为索引来获取子集
        result = ser[list(mask)]
        # 创建期望的 Series 对象
        expected = ser[mask]
        # 断言两个 Series 对象相等
        tm.assert_series_equal(result, expected)
        # 断言结果的索引与预期的相匹配
        tm.assert_index_equal(result.index, ser.index[mask])
    # 测试空 Series 的布尔索引情况
    def test_getitem_boolean_empty(self):
        # 创建一个空的 Series，数据类型为 np.int64
        ser = Series([], dtype=np.int64)
        # 设置索引的名称为 "index_name"
        ser.index.name = "index_name"
        # 使用布尔索引从空的 Series 中获取数据，此时应该返回空的 Series
        ser = ser[ser.isna()]
        # 断言索引的名称仍然为 "index_name"
        assert ser.index.name == "index_name"
        # 断言 Series 的数据类型为 np.int64

        assert ser.dtype == np.int64

        # GH#5877
        # 使用空的 Series 进行索引操作
        # 预期结果是一个空的 Series，数据类型为 object，索引为空 Index，数据类型为 "int64"
        expected = Series(dtype=object, index=Index([], dtype="int64"))
        # 对 Series 使用空的 Series 进行布尔索引操作
        result = ser[Series([], dtype=object)]
        # 断言两个 Series 是否相等
        tm.assert_series_equal(result, expected)

        # 由于布尔索引器为空或未对齐，因此操作无效
        msg = (
            r"Unalignable boolean Series provided as indexer \(index of "
            r"the boolean Series and of the indexed object do not match"
        )
        # 使用 pytest 检查是否抛出预期的 IndexingError 异常，并匹配错误消息
        with pytest.raises(IndexingError, match=msg):
            ser[Series([], dtype=bool)]

        with pytest.raises(IndexingError, match=msg):
            ser[Series([True], dtype=bool)]

    # 测试从对象中获取布尔索引的情况
    def test_getitem_boolean_object(self, string_series):
        # 使用 DataFrame 中的列进行操作

        ser = string_series
        # 创建一个布尔掩码，用于比较是否大于中位数
        mask = ser > ser.median()
        # 将布尔掩码转换为对象类型
        omask = mask.astype(object)

        # 使用布尔索引获取数据
        result = ser[omask]
        # 期望的结果是根据原始布尔掩码获取的数据
        expected = ser[mask]
        # 断言两个 Series 是否相等
        tm.assert_series_equal(result, expected)

        # 使用布尔索引设置数据
        s2 = ser.copy()
        cop = ser.copy()
        cop[omask] = 5
        s2[mask] = 5
        # 断言两个 Series 是否相等
        tm.assert_series_equal(cop, s2)

        # NaN 值会引发异常
        omask[5:10] = np.nan
        msg = "Cannot mask with non-boolean array containing NA / NaN values"
        # 使用 pytest 检查是否抛出预期的 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            ser[omask]
        with pytest.raises(ValueError, match=msg):
            ser[omask] = 5

    # 测试从日期时间 Series 中获取布尔索引的情况
    def test_getitem_boolean_dt64_copies(self):
        # GH#36210
        # 创建一个日期范围
        dti = date_range("2016-01-01", periods=4, tz="US/Pacific")
        # 创建布尔键
        key = np.array([True, True, False, False])

        # 使用日期时间数据创建 Series
        ser = Series(dti._data)

        # 使用布尔索引获取数据
        res = ser[key]
        # 断言结果中的值是否不共享基础数据
        assert res._values._ndarray.base is None

        # 用数值索引的情况进行比较
        ser2 = Series(range(4))
        res2 = ser2[key]
        # 断言结果中的值是否不共享基础数据
        assert res2._values.base is None

    # 测试日期时间 Series 中的布尔索引的极端情况
    def test_getitem_boolean_corner(self, datetime_series):
        ts = datetime_series
        # 创建一个偏移后的布尔掩码
        mask_shifted = ts.shift(1, freq=BDay()) > ts.median()

        msg = (
            r"Unalignable boolean Series provided as indexer \(index of "
            r"the boolean Series and of the indexed object do not match"
        )
        # 使用 pytest 检查是否抛出预期的 IndexingError 异常，并匹配错误消息
        with pytest.raises(IndexingError, match=msg):
            ts[mask_shifted]

        with pytest.raises(IndexingError, match=msg):
            ts.loc[mask_shifted]

    # 测试对字符串 Series 使用不同排序的布尔索引的情况
    def test_getitem_boolean_different_order(self, string_series):
        # 对字符串 Series 进行排序
        ordered = string_series.sort_values()

        # 使用排序后的 Series 进行布尔索引选择
        sel = string_series[ordered > 0]
        # 期望的结果是根据原始 Series 的条件选择
        exp = string_series[string_series > 0]
        # 断言两个 Series 是否相等
        tm.assert_series_equal(sel, exp)
    # 定义一个测试函数，测试从日期范围中获取布尔索引并保留频率的操作
    def test_getitem_boolean_contiguous_preserve_freq(self):
        # 创建一个日期范围对象 rng，从 "2000-01-01" 到 "2000-03-01"，频率为工作日（Business day）
        rng = date_range("1/1/2000", "3/1/2000", freq="B")

        # 创建一个长度与日期范围 rng 相同的布尔数组 mask，初始值为 False
        mask = np.zeros(len(rng), dtype=bool)
        # 将索引为 10 到 19 的位置设为 True
        mask[10:20] = True

        # 使用布尔数组 mask 对日期范围 rng 进行索引，得到新的日期范围 masked
        masked = rng[mask]
        # 从日期范围 rng 中取出索引为 10 到 19 的子集作为期望的结果 expected
        expected = rng[10:20]
        # 断言期望的结果 expected 的频率与原始日期范围 rng 的频率相同
        assert expected.freq == rng.freq
        # 使用测试工具函数 tm.assert_index_equal 检查 masked 和 expected 是否相等
        tm.assert_index_equal(masked, expected)

        # 将索引为 22 的位置设为 True
        mask[22] = True
        # 再次使用 mask 对日期范围 rng 进行索引，得到新的日期范围 masked
        masked = rng[mask]
        # 断言 masked 的频率为 None
        assert masked.freq is None
class TestGetitemCallable:
    def test_getitem_callable(self):
        # 创建一个包含四个元素的 Series 对象，指定索引为 ['A', 'B', 'C', 'D']
        ser = Series(4, index=list("ABCD"))
        # 使用 lambda 函数从 Series 中选择包含 "A" 的元素
        result = ser[lambda x: "A"]
        # 断言选择的结果与 ser.loc["A"] 相等
        assert result == ser.loc["A"]

        # 使用 lambda 函数从 Series 中选择包含 ["A", "B"] 的元素
        result = ser[lambda x: ["A", "B"]]
        # 预期结果是 ser.loc[["A", "B"]]
        expected = ser.loc[["A", "B"]]
        # 使用测试工具函数检查结果是否相等
        tm.assert_series_equal(result, expected)

        # 使用 lambda 函数从 Series 中选择满足条件 [True, False, True, True] 的元素
        result = ser[lambda x: [True, False, True, True]]
        # 预期结果是 ser.iloc[[0, 2, 3]]
        expected = ser.iloc[[0, 2, 3]]
        # 使用测试工具函数检查结果是否相等
        tm.assert_series_equal(result, expected)


def test_getitem_generator(string_series):
    # 创建一个生成器 gen，包含 string_series 中大于 0 的元素
    gen = (x > 0 for x in string_series)
    # 使用生成器 gen 从 string_series 中选择符合条件的元素
    result = string_series[gen]
    # 使用 iter 函数创建迭代器，但传入的表达式语法有误
    result2 = string_series[iter(string_series > 0)]
    # 预期结果是 string_series[string_series > 0]
    expected = string_series[string_series > 0]
    # 使用测试工具函数检查结果是否相等
    tm.assert_series_equal(result, expected)
    tm.assert_series_equal(result2, expected)


@pytest.mark.parametrize(
    "data",
    [
        [0, 1],  # 创建一个包含 [0, 1] 的列表
        date_range("2012-01-01", periods=2),  # 创建一个包含两个日期的日期范围
        date_range("2012-01-01", periods=2, tz="CET"),  # 创建一个带有时区的日期范围
    ],
)
def test_getitem_ndim_deprecated(data):
    # 创建一个 Series 对象，使用不推荐使用的多维索引
    series = Series(data)
    # 使用 pytest 断言引发 ValueError 异常，并匹配错误信息 "Multi-dimensional indexing"
    with pytest.raises(ValueError, match="Multi-dimensional indexing"):
        series[:, None]


def test_getitem_multilevel_scalar_slice_not_implemented(
    multiindex_year_month_day_dataframe_random_data,
):
    # 暂不实现此功能
    df = multiindex_year_month_day_dataframe_random_data
    # 从 DataFrame 中选择 'A' 列，返回一个 Series 对象
    ser = df["A"]

    # 设置预期的错误消息格式
    msg = r"\(2000, slice\(3, 4, None\)\)"
    # 使用 pytest 断言引发 TypeError 异常，并匹配预期的错误消息格式
    with pytest.raises(TypeError, match=msg):
        ser[2000, 3:4]


def test_getitem_dataframe_raises():
    # 创建一个包含整数的列表 rng
    rng = list(range(10))
    # 创建一个包含整数 10 的 Series 对象，指定索引为 rng
    ser = Series(10, index=rng)
    # 创建一个 DataFrame 对象，使用 rng 作为数据和索引
    df = DataFrame(rng, index=rng)
    # 设置预期的错误消息
    msg = (
        "Indexing a Series with DataFrame is not supported, "
        "use the appropriate DataFrame column"
    )
    # 使用 pytest 断言引发 TypeError 异常，并匹配预期的错误消息
    with pytest.raises(TypeError, match=msg):
        ser[df > 5]


def test_getitem_assignment_series_alignment():
    # https://github.com/pandas-dev/pandas/issues/37427
    # 使用 getitem 时，当使用 Series 进行赋值时，不会首先对齐
    ser = Series(range(10))
    # 创建一个包含索引数组的 idx
    idx = np.array([2, 4, 9])
    # 使用 idx 来为 ser 中的元素赋值，使用另一个 Series 对象
    ser[idx] = Series([10, 11, 12])
    # 设置预期的 Series 对象
    expected = Series([0, 1, 10, 3, 11, 5, 6, 7, 8, 12])
    # 使用测试工具函数检查结果是否相等
    tm.assert_series_equal(ser, expected)


def test_getitem_duplicate_index_mistyped_key_raises_keyerror():
    # GH#29189 float_index.get_loc(None) 应该引发 KeyError，而不是 TypeError
    # 创建一个带有浮点数索引的 Series 对象
    ser = Series([2, 5, 6, 8], index=[2.0, 4.0, 4.0, 5.0])
    # 使用 pytest 断言引发 KeyError 异常，并匹配 "None" 错误消息
    with pytest.raises(KeyError, match="None"):
        ser[None]

    # 使用 pytest 断言引发 KeyError 异常，并匹配 "None" 错误消息
    with pytest.raises(KeyError, match="None"):
        ser.index.get_loc(None)

    # 使用 pytest 断言引发 KeyError 异常，并匹配 "None" 错误消息
    with pytest.raises(KeyError, match="None"):
        ser.index._engine.get_loc(None)


def test_getitem_1tuple_slice_without_multiindex():
    # 创建一个包含五个元素的 Series 对象
    ser = Series(range(5))
    # 创建一个元组 key，包含一个 slice 对象
    key = (slice(3),)

    # 使用 key 从 Series 中选择元素
    result = ser[key]
    # 预期结果是 ser[key[0]]
    expected = ser[key[0]]
    # 使用测试工具函数检查结果是否相等
    tm.assert_series_equal(result, expected)


def test_getitem_preserve_name(datetime_series):
    # 使用 getitem 选择 datetime_series 中大于 0 的元素
    result = datetime_series[datetime_series > 0]
    # 断言结果的名称与原始 Series 对象的名称相同
    assert result.name == datetime_series.name

    # 使用 getitem 选择 datetime_series 中索引从 5 到 10 的元素
    result = datetime_series[5:10]
    # 断言检查：验证 result 对象的 name 属性与 datetime_series 对象的 name 属性是否相等
    assert result.name == datetime_series.name
# 测试函数：测试带整数标签的 Series 的索引操作
def test_getitem_with_integer_labels():
    # 创建一个 Series 对象，使用随机数生成器生成标准正态分布的数据，索引为偶数序列
    ser = Series(
        np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2))
    )
    # 设定索引列表和数组，用于测试索引操作
    inds = [0, 2, 5, 7, 8]
    arr_inds = np.array([0, 2, 5, 7, 8])
    # 使用 pytest 验证索引不存在时是否会抛出 KeyError 异常，并匹配特定错误信息
    with pytest.raises(KeyError, match="not in index"):
        ser[inds]

    with pytest.raises(KeyError, match="not in index"):
        ser[arr_inds]


# 测试函数：测试缺失索引情况下的索引操作
def test_getitem_missing(datetime_series):
    # 计算出一个过去的日期作为索引，用于测试缺失索引时的异常情况
    d = datetime_series.index[0] - BDay()
    # 准备错误信息的正则表达式匹配模式
    msg = r"Timestamp\('1999-12-31 00:00:00'\)"
    # 使用 pytest 验证缺失索引时是否会抛出 KeyError 异常，并匹配特定错误信息
    with pytest.raises(KeyError, match=msg):
        datetime_series[d]


# 测试函数：测试复杂索引列表的索引操作
def test_getitem_fancy(string_series, object_series):
    # 准备错误信息的正则表达式匹配模式
    msg = r"None of \[Index\(\[1, 2, 3\], dtype='int(32|64)'\)\] are in the \[index\]"
    # 使用 pytest 验证复杂索引列表时是否会抛出 KeyError 异常，并匹配特定错误信息
    with pytest.raises(KeyError, match=msg):
        string_series[[1, 2, 3]]
    with pytest.raises(KeyError, match=msg):
        object_series[[1, 2, 3]]


# 测试函数：测试索引未排序且存在重复的情况
def test_getitem_unordered_dup():
    # 创建一个带有重复索引的 Series 对象
    obj = Series(range(5), index=["c", "a", "a", "b", "b"])
    # 使用 assert 检查索引为"c"的元素是否为标量，并且值为0
    assert is_scalar(obj["c"])
    assert obj["c"] == 0


# 测试函数：测试索引存在重复的情况
def test_getitem_dups():
    # 创建一个带有重复索引的 Series 对象
    ser = Series(range(5), index=["A", "A", "B", "C", "C"], dtype=np.int64)
    # 准备预期的结果 Series 对象
    expected = Series([3, 4], index=["C", "C"], dtype=np.int64)
    # 进行索引操作并验证结果是否符合预期
    result = ser["C"]
    tm.assert_series_equal(result, expected)


# 测试函数：测试分类变量索引为字符串的情况
def test_getitem_categorical_str():
    # 使用分类变量作为索引创建一个 Series 对象
    ser = Series(range(5), index=Categorical(["a", "b", "c", "a", "b"]))
    # 进行索引操作并验证结果是否符合预期
    result = ser["a"]
    expected = ser.iloc[[0, 3]]
    tm.assert_series_equal(result, expected)


# 测试函数：测试对索引非唯一排序的 Series 对象进行切片操作
def test_slice_can_reorder_not_uniquely_indexed():
    # 创建一个索引中存在重复的 Series 对象，并进行反向切片操作
    ser = Series(1, index=["a", "a", "b", "b", "c"])
    ser[::-1]  # it works!


# 参数化测试函数：测试具有重复索引的 Series 对象，使用位置索引器进行索引操作
@pytest.mark.parametrize("index_vals", ["aabcd", "aadcb"])
def test_duplicated_index_getitem_positional_indexer(index_vals):
    # 使用给定的索引值列表创建一个带有重复索引的 Series 对象
    s = Series(range(5), index=list(index_vals))
    # 使用 pytest 验证索引为3时是否会抛出 KeyError 异常，并匹配特定错误信息
    with pytest.raises(KeyError, match="^3$"):
        s[3]


# 测试类：测试已弃用的索引器
class TestGetitemDeprecatedIndexers:
    # 参数化测试函数：测试使用字典和集合作为索引器时的异常情况
    @pytest.mark.parametrize("key", [{1}, {1: 1}])
    def test_getitem_dict_and_set_deprecated(self, key):
        # 创建一个 Series 对象
        ser = Series([1, 2, 3])
        # 使用 pytest 验证使用字典和集合作为索引器时是否会抛出 TypeError 异常
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            ser[key]

    # 参数化测试函数：测试使用字典和集合作为索引器设置值时的异常情况
    @pytest.mark.parametrize("key", [{1}, {1: 1}])
    def test_setitem_dict_and_set_disallowed(self, key):
        # 创建一个 Series 对象
        ser = Series([1, 2, 3])
        # 使用 pytest 验证使用字典和集合作为索引器设置值时是否会抛出 TypeError 异常
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            ser[key] = 1
```