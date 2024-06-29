# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_datetime.py`

```
import datetime as dt  # 导入 datetime 模块，并用别名 dt 引用

from datetime import date  # 从 datetime 模块中导入 date 类

import numpy as np  # 导入 NumPy 库，并用别名 np 引用
import pytest  # 导入 pytest 测试框架

from pandas.compat.numpy import np_long  # 从 pandas 中的 compat 模块导入 np_long

import pandas as pd  # 导入 Pandas 库，并用别名 pd 引用
from pandas import (  # 从 Pandas 中导入多个对象
    DataFrame,  # 数据帧对象
    DatetimeIndex,  # 日期时间索引对象
    Index,  # 索引对象
    Timestamp,  # 时间戳对象
    date_range,  # 日期范围生成函数
    offsets,  # 时间偏移对象
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

class TestDatetimeIndex:  # 定义测试类 TestDatetimeIndex
    def test_is_(self):  # 定义测试方法 test_is_
        dti = date_range(start="1/1/2005", end="12/1/2005", freq="ME")  # 创建一个月末频率的日期时间索引
        assert dti.is_(dti)  # 断言判断 dti 是否为其本身
        assert dti.is_(dti.view())  # 断言判断 dti 是否为其视图
        assert not dti.is_(dti.copy())  # 断言判断 dti 是否为其副本

    def test_time_overflow_for_32bit_machines(self):  # 定义测试方法 test_time_overflow_for_32bit_machines
        # GH8943.  On some machines NumPy defaults to np.int32 (for example,
        # 32-bit Linux machines).  In the function _generate_regular_range
        # found in tseries/index.py, `periods` gets multiplied by `strides`
        # (which has value 1e9) and since the max value for np.int32 is ~2e9,
        # and since those machines won't promote np.int32 to np.int64, we get
        # overflow.
        periods = np_long(1000)  # 创建一个 np_long 对象 periods，值为 1000

        idx1 = date_range(start="2000", periods=periods, freq="s")  # 创建一个秒频率的日期时间索引 idx1
        assert len(idx1) == periods  # 断言判断 idx1 的长度是否等于 periods

        idx2 = date_range(end="2000", periods=periods, freq="s")  # 创建一个从 end="2000" 开始的秒频率的日期时间索引 idx2
        assert len(idx2) == periods  # 断言判断 idx2 的长度是否等于 periods

    def test_nat(self):  # 定义测试方法 test_nat
        assert DatetimeIndex([np.nan])[0] is pd.NaT  # 断言判断 DatetimeIndex 中的第一个元素是否为 NaT

    def test_week_of_month_frequency(self):  # 定义测试方法 test_week_of_month_frequency
        # GH 5348: "ValueError: Could not evaluate WOM-1SUN" shouldn't raise
        d1 = date(2002, 9, 1)  # 创建日期对象 d1
        d2 = date(2013, 10, 27)  # 创建日期对象 d2
        d3 = date(2012, 9, 30)  # 创建日期对象 d3
        idx1 = DatetimeIndex([d1, d2])  # 创建包含 d1 和 d2 的日期时间索引 idx1
        idx2 = DatetimeIndex([d3])  # 创建包含 d3 的日期时间索引 idx2
        result_append = idx1.append(idx2)  # 将 idx2 追加到 idx1，返回结果存储在 result_append 中
        expected = DatetimeIndex([d1, d2, d3])  # 创建预期的日期时间索引 expected
        tm.assert_index_equal(result_append, expected)  # 使用测试模块 tm 断言判断 result_append 是否等于 expected
        result_union = idx1.union(idx2)  # 对 idx1 和 idx2 执行并集操作，结果存储在 result_union 中
        expected = DatetimeIndex([d1, d3, d2])  # 更新预期的日期时间索引 expected
        tm.assert_index_equal(result_union, expected)  # 使用测试模块 tm 断言判断 result_union 是否等于 expected

    def test_append_nondatetimeindex(self):  # 定义测试方法 test_append_nondatetimeindex
        rng = date_range("1/1/2000", periods=10)  # 创建一个日期范围对象 rng
        idx = Index(["a", "b", "c", "d"])  # 创建一个索引对象 idx

        result = rng.append(idx)  # 将 idx 追加到 rng，结果存储在 result 中
        assert isinstance(result[0], Timestamp)  # 断言判断 result 的第一个元素是否为 Timestamp 对象

    def test_misc_coverage(self):  # 定义测试方法 test_misc_coverage
        rng = date_range("1/1/2000", periods=5)  # 创建一个包含 5 个日期的日期范围对象 rng
        result = rng.groupby(rng.day)  # 对 rng 根据 day 进行分组，结果存储在 result 中
        assert isinstance(next(iter(result.values()))[0], Timestamp)  # 断言判断 result 的第一个分组值的第一个元素是否为 Timestamp 对象

    # TODO: belongs in frame groupby tests?
    def test_groupby_function_tuple_1677(self):  # 定义测试方法 test_groupby_function_tuple_1677
        df = DataFrame(  # 创建一个 DataFrame 对象 df
            np.random.default_rng(2).random(100),  # 生成一个随机数组成的 Series 对象，长度为 100
            index=date_range("1/1/2000", periods=100),  # 创建一个包含 100 个日期的日期范围作为索引
        )
        monthly_group = df.groupby(lambda x: (x.year, x.month))  # 根据年份和月份对 df 进行分组，结果存储在 monthly_group 中

        result = monthly_group.mean()  # 计算分组后的均值，结果存储在 result 中
        assert isinstance(result.index[0], tuple)  # 断言判断 result 的第一个索引是否为元组类型

    def assert_index_parameters(self, index):  # 定义辅助方法 assert_index_parameters，参数为 index
        assert index.freq == "40960ns"  # 断言判断 index 的频率是否为 "40960ns"
        assert index.inferred_freq == "40960ns"  # 断言判断 index 推断的频率是否为 "40960ns"
    def test_ns_index(self):
        # 设置样本数为400
        nsamples = 400
        # 计算每个样本的时间间隔（以纳秒为单位）
        ns = int(1e9 / 24414)
        # 定义起始日期时间
        dtstart = np.datetime64("2012-09-20T00:00:00")

        # 根据起始日期时间和时间间隔创建日期时间序列
        dt = dtstart + np.arange(nsamples) * np.timedelta64(ns, "ns")
        # 根据时间间隔创建频率对象
        freq = ns * offsets.Nano()
        # 创建日期时间索引对象，并命名为"time"
        index = DatetimeIndex(dt, freq=freq, name="time")
        # 调用辅助函数验证索引参数
        self.assert_index_parameters(index)

        # 根据索引的首尾元素创建新的日期时间序列
        new_index = date_range(start=index[0], end=index[-1], freq=index.freq)
        # 调用辅助函数验证索引参数
        self.assert_index_parameters(new_index)

    def test_asarray_tz_naive(self):
        # 测试对无时区信息日期范围的数组化操作
        # 不应该产生警告信息
        idx = date_range("2000", periods=2)
        # 默认情况下，转换为M8[ns]类型的numpy数组
        result = np.asarray(idx)

        # 期望的结果数组
        expected = np.array(["2000-01-01", "2000-01-02"], dtype="M8[ns]")
        # 验证numpy数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 可选的，转换为object类型的数组
        result = np.asarray(idx, dtype=object)

        # 期望的结果数组，包含Timestamp对象
        expected = np.array([Timestamp("2000-01-01"), Timestamp("2000-01-02")])
        # 验证numpy数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    def test_asarray_tz_aware(self):
        # 测试对带有时区信息日期范围的数组化操作
        tz = "US/Central"
        idx = date_range("2000", periods=2, tz=tz)
        # 期望的结果数组
        expected = np.array(["2000-01-01T06", "2000-01-02T06"], dtype="M8[ns]")
        # 转换为datetime64[ns]类型的numpy数组
        result = np.asarray(idx, dtype="datetime64[ns]")

        # 验证numpy数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 旧的行为，不应产生警告
        result = np.asarray(idx, dtype="M8[ns]")

        # 验证numpy数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 未来的行为，不应产生警告
        expected = np.array(
            [Timestamp("2000-01-01", tz=tz), Timestamp("2000-01-02", tz=tz)]
        )
        # 转换为object类型的numpy数组
        result = np.asarray(idx, dtype=object)

        # 验证numpy数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    def test_CBH_deprecated(self):
        # 测试"CBH"频率在未来版本中将被移除的警告
        msg = "'CBH' is deprecated and will be removed in a future version."

        # 断言产生FutureWarning警告，并匹配预期的警告信息
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 创建"CBH"频率的日期范围
            expected = date_range(
                dt.datetime(2022, 12, 11), dt.datetime(2022, 12, 13), freq="CBH"
            )
        # 创建日期时间索引对象，使用"cbh"频率
        result = DatetimeIndex(
            [
                "2022-12-12 09:00:00",
                "2022-12-12 10:00:00",
                "2022-12-12 11:00:00",
                "2022-12-12 12:00:00",
                "2022-12-12 13:00:00",
                "2022-12-12 14:00:00",
                "2022-12-12 15:00:00",
                "2022-12-12 16:00:00",
            ],
            dtype="datetime64[ns]",
            freq="cbh",
        )

        # 验证日期时间索引对象是否等于预期值
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("freq", ["2BM", "1bm", "2BQ", "1BQ-MAR", "2BY-JUN", "1by"])
    def test_BM_BQ_BY_raises(self, freq):
        # 参数化测试，检查无效频率的异常抛出情况
        msg = f"Invalid frequency: {freq}"

        # 断言抛出ValueError异常，并匹配预期的错误信息
        with pytest.raises(ValueError, match=msg):
            # 尝试创建使用无效频率的日期范围
            date_range(start="2016-02-21", end="2016-08-21", freq=freq)
    # 定义一个测试方法，测试在给定频率时是否会引发 ValueError 异常
    def test_BA_BAS_raises(self, freq):
        # 构建错误消息，指出无效的频率
        msg = f"Invalid frequency: {freq}"

        # 使用 pytest 框架的 raises 方法验证在调用 date_range 函数时是否会抛出 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            # 调用 date_range 函数，预期会因为无效的频率而抛出 ValueError 异常
            date_range(start="2016-02-21", end="2016-08-21", freq=freq)
```