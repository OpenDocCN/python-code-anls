# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\test_partial_slicing.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试

from pandas import (  # 从 pandas 库中导入多个模块和类
    DataFrame,  # 数据帧类，用于处理表格数据
    PeriodIndex,  # 时期索引类，用于处理时间序列数据
    Series,  # 系列类，用于处理一维数据结构
    date_range,  # 日期范围生成函数
    period_range,  # 时期范围生成函数
)
import pandas._testing as tm  # 导入 pandas 内部测试工具模块

class TestPeriodIndex:
    def test_getitem_periodindex_duplicates_string_slice(self):
        # monotonic
        idx = PeriodIndex([2000, 2007, 2007, 2009, 2009], freq="Y-JUN")  # 创建一个具有重复条目的时期索引对象
        ts = Series(np.random.default_rng(2).standard_normal(len(idx)), index=idx)  # 使用随机数生成数据的系列对象，索引为 idx
        original = ts.copy()  # 备份原始系列对象

        result = ts["2007"]  # 从系列中选择指定时期的数据
        expected = ts[1:3]  # 从系列中选择切片区间的数据
        tm.assert_series_equal(result, expected)  # 断言两个系列对象是否相等
        result[:] = 1  # 将 result 中所有元素设置为 1
        tm.assert_series_equal(ts, original)  # 断言修改后的系列与原始系列相等

        # not monotonic
        idx = PeriodIndex([2000, 2007, 2007, 2009, 2007], freq="Y-JUN")  # 创建一个非单调的时期索引对象
        ts = Series(np.random.default_rng(2).standard_normal(len(idx)), index=idx)  # 使用随机数生成数据的系列对象，索引为 idx

        result = ts["2007"]  # 从系列中选择指定时期的数据
        expected = ts[idx == "2007"]  # 使用条件选择从系列中选择符合条件的数据
        tm.assert_series_equal(result, expected)  # 断言两个系列对象是否相等

    def test_getitem_periodindex_quarter_string(self):
        pi = PeriodIndex(["2Q05", "3Q05", "4Q05", "1Q06", "2Q06"], freq="Q")  # 创建一个以季度为频率的时期索引对象
        ser = Series(np.random.default_rng(2).random(len(pi)), index=pi).cumsum()  # 使用随机数生成数据的系列对象，并对其进行累计求和操作
        # Todo: fix these accessors!
        assert ser["05Q4"] == ser.iloc[2]  # 断言根据字符串进行索引的结果与 iloc 方法得到的结果相等

    def test_pindex_slice_index(self):
        pi = period_range(start="1/1/10", end="12/31/12", freq="M")  # 创建一个时间段范围的时期索引对象
        s = Series(np.random.default_rng(2).random(len(pi)), index=pi)  # 使用随机数生成数据的系列对象，索引为 pi
        res = s["2010"]  # 根据年份选择系列中的数据
        exp = s[0:12]  # 选择索引范围为 0 到 12 的数据
        tm.assert_series_equal(res, exp)  # 断言两个系列对象是否相等
        res = s["2011"]  # 根据年份选择系列中的数据
        exp = s[12:24]  # 选择索引范围为 12 到 24 的数据
        tm.assert_series_equal(res, exp)  # 断言两个系列对象是否相等

    @pytest.mark.parametrize("make_range", [date_range, period_range])
    def test_range_slice_day(self, make_range):
        # GH#6716
        idx = make_range(start="2013/01/01", freq="D", periods=400)  # 使用指定的生成函数生成日期或时期范围的索引对象

        msg = "slice indices must be integers or None or have an __index__ method"  # 错误消息字符串
        # slices against index should raise IndexError
        values = [
            "2014",
            "2013/02",
            "2013/01/02",
            "2013/02/01 9H",
            "2013/02/01 09:00",
        ]
        for v in values:
            with pytest.raises(TypeError, match=msg):  # 使用 pytest 检查对索引的切片操作是否引发 TypeError 异常
                idx[v:]

        s = Series(np.random.default_rng(2).random(len(idx)), index=idx)  # 使用随机数生成数据的系列对象，索引为 idx

        tm.assert_series_equal(s["2013/01/02":], s[1:])  # 断言根据日期或时期字符串进行的切片操作与基于整数索引的切片操作结果相等
        tm.assert_series_equal(s["2013/01/02":"2013/01/05"], s[1:5])  # 断言根据日期或时期字符串进行的切片操作与基于整数索引的切片操作结果相等
        tm.assert_series_equal(s["2013/02":], s[31:])  # 断言根据日期或时期字符串进行的切片操作与基于整数索引的切片操作结果相等
        tm.assert_series_equal(s["2014":], s[365:])  # 断言根据日期或时期字符串进行的切片操作与基于整数索引的切片操作结果相等

        invalid = ["2013/02/01 9H", "2013/02/01 09:00"]
        for v in invalid:
            with pytest.raises(TypeError, match=msg):  # 使用 pytest 检查对索引的切片操作是否引发 TypeError 异常
                idx[v:]
    def test_range_slice_seconds(self, make_range):
        # 定义一个测试方法，测试时间范围切片的行为，使用make_range生成时间索引
        idx = make_range(start="2013/01/01 09:00:00", freq="s", periods=4000)
        # 错误消息，用于断言切片索引必须是整数、None或具有__index__方法
        msg = "slice indices must be integers or None or have an __index__ method"

        # 准备多个切片索引进行测试，预期每个切片操作会抛出TypeError异常，且异常消息匹配msg
        values = [
            "2014",
            "2013/02",
            "2013/01/02",
            "2013/02/01 9H",
            "2013/02/01 09:00",
        ]
        for v in values:
            with pytest.raises(TypeError, match=msg):
                idx[v:]

        # 创建一个Series对象，使用随机数填充，指定索引为idx
        s = Series(np.random.default_rng(2).random(len(idx)), index=idx)

        # 使用tm.assert_series_equal断言不同时间范围的切片操作后的Series对象相等
        tm.assert_series_equal(s["2013/01/01 09:05":"2013/01/01 09:10"], s[300:660])
        tm.assert_series_equal(s["2013/01/01 10:00":"2013/01/01 10:05"], s[3600:3960])
        tm.assert_series_equal(s["2013/01/01 10H":], s[3600:])
        tm.assert_series_equal(s[:"2013/01/01 09:30"], s[:1860])
        for d in ["2013/01/01", "2013/01", "2013"]:
            tm.assert_series_equal(s[d:], s)

    @pytest.mark.parametrize("make_range", [date_range, period_range])
    def test_range_slice_outofbounds(self, make_range):
        # GH#5407，测试超出边界的时间范围切片操作
        idx = make_range(start="2013/10/01", freq="D", periods=10)

        # 创建一个DataFrame对象，包含名为"units"的列，索引为idx
        df = DataFrame({"units": [100 + i for i in range(10)]}, index=idx)
        # 创建一个空的DataFrame对象，索引与idx的前0个元素相同，列名为"units"，类型为int64
        empty = DataFrame(index=idx[:0], columns=["units"])
        empty["units"] = empty["units"].astype("int64")

        # 使用tm.assert_frame_equal断言不同时间范围的切片操作后的DataFrame对象相等
        tm.assert_frame_equal(df["2013/09/01":"2013/09/30"], empty)
        tm.assert_frame_equal(df["2013/09/30":"2013/10/02"], df.iloc[:2])
        tm.assert_frame_equal(df["2013/10/01":"2013/10/02"], df.iloc[:2])
        tm.assert_frame_equal(df["2013/10/02":"2013/09/30"], empty)
        tm.assert_frame_equal(df["2013/10/15":"2013/10/17"], empty)
        tm.assert_frame_equal(df["2013-06":"2013-09"], empty)
        tm.assert_frame_equal(df["2013-11":"2013-12"], empty)

    @pytest.mark.parametrize("make_range", [date_range, period_range])
    def test_maybe_cast_slice_bound(self, make_range, frame_or_series):
        # 使用 make_range 函数创建一个时间范围索引 idx，起始于 "2013/10/01"，频率为每日，共 10 个周期
        idx = make_range(start="2013/10/01", freq="D", periods=10)

        # 创建一个 DataFrame 对象 obj，包含一个名为 "units" 的列，其值为 100 到 109
        obj = DataFrame({"units": [100 + i for i in range(10)]}, index=idx)
        # 使用 tm.get_obj 函数获取 obj 的处理结果，frame_or_series 是输入参数
        obj = tm.get_obj(obj, frame_or_series)

        # 创建错误消息，指示在类型为 idx.__name__ 的对象上使用字符串类型的索引器 "foo" 会导致 TypeError
        msg = (
            f"cannot do slice indexing on {type(idx).__name__} with "
            r"these indexers \[foo\] of type str"
        )

        # 检查下层调用是否在预期位置引发异常
        with pytest.raises(TypeError, match=msg):
            # 调用 idx 对象的 _maybe_cast_slice_bound 方法，期望引发 TypeError 异常，且异常消息符合预期
            idx._maybe_cast_slice_bound("foo", "left")
        with pytest.raises(TypeError, match=msg):
            # 调用 idx 对象的 get_slice_bound 方法，期望引发 TypeError 异常，且异常消息符合预期
            idx.get_slice_bound("foo", "left")

        with pytest.raises(TypeError, match=msg):
            # 尝试对 obj 执行切片索引，使用非法字符串 "foo"，期望引发 TypeError 异常，且异常消息符合预期
            obj["2013/09/30":"foo"]
        with pytest.raises(TypeError, match=msg):
            # 尝试对 obj 执行切片索引，使用非法字符串 "foo"，期望引发 TypeError 异常，且异常消息符合预期
            obj["foo":"2013/09/30"]
        with pytest.raises(TypeError, match=msg):
            # 尝试对 obj 执行 loc 定位，使用非法字符串 "foo"，期望引发 TypeError 异常，且异常消息符合预期
            obj.loc["2013/09/30":"foo"]
        with pytest.raises(TypeError, match=msg):
            # 尝试对 obj 执行 loc 定位，使用非法字符串 "foo"，期望引发 TypeError 异常，且异常消息符合预期
            obj.loc["foo":"2013/09/30"]

    def test_partial_slice_doesnt_require_monotonicity(self):
        # 查看 DatetimeIndex 测试的同名部分
        dti = date_range("2014-01-01", periods=30, freq="30D")
        pi = dti.to_period("D")

        # 创建一个 Series 对象 ser_montonic，包含 0 到 29 的整数值，索引为 pi
        ser_montonic = Series(np.arange(30), index=pi)

        # 创建一个序列 shuffler，包含序列索引的乱序排列
        shuffler = list(range(0, 30, 2)) + list(range(1, 31, 2))
        # 使用 shuffler 对 ser_montonic 进行切片选择，得到新的 Series 对象 ser
        ser = ser_montonic.iloc[shuffler]
        # 获取新的索引 nidx
        nidx = ser.index

        # 手动确定 year==2014 的位置
        indexer_2014 = np.array(
            [0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20], dtype=np.intp
        )
        # 断言 nidx 中指定位置的年份都为 2014
        assert (nidx[indexer_2014].year == 2014).all()
        # 断言 nidx 中非指定位置的年份没有都为 2014
        assert not (nidx[~indexer_2014].year == 2014).any()

        # 获取 "2014" 在 nidx 中的位置，返回结果为 result
        result = nidx.get_loc("2014")
        # 使用 tm.assert_numpy_array_equal 断言 result 与 indexer_2014 相等
        tm.assert_numpy_array_equal(result, indexer_2014)

        # 获取 ser 中 "2014" 的数据，预期结果为 expected
        expected = ser.iloc[indexer_2014]
        # 执行 ser 的 loc["2014"] 操作，返回结果为 result
        result = ser.loc["2014"]
        # 使用 tm.assert_series_equal 断言 result 与 expected 相等
        tm.assert_series_equal(result, expected)

        # 执行 ser["2014"] 操作，返回结果为 result
        result = ser["2014"]
        # 使用 tm.assert_series_equal 断言 result 与 expected 相等
        tm.assert_series_equal(result, expected)

        # 手动确定 ser.index 中年份为 2015 且月份为 5 的位置
        indexer_may2015 = np.array([23], dtype=np.intp)
        # 断言 nidx 中指定位置的年份为 2015 且月份为 5
        assert nidx[23].year == 2015 and nidx[23].month == 5

        # 获取 "May 2015" 在 nidx 中的位置，返回结果为 result
        result = nidx.get_loc("May 2015")
        # 使用 tm.assert_numpy_array_equal 断言 result 与 indexer_may2015 相等
        tm.assert_numpy_array_equal(result, indexer_may2015)

        # 获取 ser 中 "May 2015" 的数据，预期结果为 expected
        expected = ser.iloc[indexer_may2015]
        # 执行 ser 的 loc["May 2015"] 操作，返回结果为 result
        result = ser.loc["May 2015"]
        # 使用 tm.assert_series_equal 断言 result 与 expected 相等
        tm.assert_series_equal(result, expected)

        # 执行 ser["May 2015"] 操作，返回结果为 result
        result = ser["May 2015"]
        # 使用 tm.assert_series_equal 断言 result 与 expected 相等
        tm.assert_series_equal(result, expected)
```