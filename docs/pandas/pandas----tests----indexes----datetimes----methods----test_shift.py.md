# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_shift.py`

```
# 从 datetime 模块中导入 datetime 类
from datetime import datetime
# 导入 zoneinfo 模块
import zoneinfo

# 导入 pytest 测试框架
import pytest

# 导入 pandas 库，并从中导入 NullFrequencyError 错误类
from pandas.errors import NullFrequencyError

# 导入 pandas 库，并使用 pd 别名
import pandas as pd
# 从 pandas 库中导入 DatetimeIndex、Series、date_range 函数
from pandas import (
    DatetimeIndex,
    Series,
    date_range,
)
# 导入 pandas._testing 模块，并使用 tm 别名
import pandas._testing as tm

# 定义一个测试类 TestDatetimeIndexShift
class TestDatetimeIndexShift:
    # -------------------------------------------------------------
    # DatetimeIndex.shift is used in integer addition
    # DatetimeIndex.shift 方法用于整数加法运算

    # 定义测试方法 test_dti_shift_tzaware，使用 tz_naive_fixture 和 unit 作为参数
    def test_dti_shift_tzaware(self, tz_naive_fixture, unit):
        # GH#9903
        # 获取时区信息
        tz = tz_naive_fixture
        # 创建一个空的带单位的 DatetimeIndex 对象 idx，名称为 "xxx"，时区为 tz
        idx = DatetimeIndex([], name="xxx", tz=tz).as_unit(unit)
        # 断言 idx 同于 idx 自身在 freq="h" 下移动 0 个单位后的结果
        tm.assert_index_equal(idx.shift(0, freq="h"), idx)
        # 断言 idx 同于 idx 自身在 freq="h" 下移动 3 个单位后的结果
        tm.assert_index_equal(idx.shift(3, freq="h"), idx)

        # 创建一个带有时间戳的 DatetimeIndex 对象 idx，时区为 tz，频率为 "h"
        idx = DatetimeIndex(
            ["2011-01-01 10:00", "2011-01-01 11:00", "2011-01-01 12:00"],
            name="xxx",
            tz=tz,
            freq="h",
        ).as_unit(unit)
        # 断言 idx 同于 idx 自身在 freq="h" 下移动 0 个单位后的结果
        tm.assert_index_equal(idx.shift(0, freq="h"), idx)
        # 创建一个预期的 DatetimeIndex 对象 exp，时区为 tz，频率为 "h"
        exp = DatetimeIndex(
            ["2011-01-01 13:00", "2011-01-01 14:00", "2011-01-01 15:00"],
            name="xxx",
            tz=tz,
            freq="h",
        ).as_unit(unit)
        # 断言 idx 同于 idx 自身在 freq="h" 下移动 3 个单位后的结果为 exp
        tm.assert_index_equal(idx.shift(3, freq="h"), exp)
        # 创建一个预期的 DatetimeIndex 对象 exp，时区为 tz，频率为 "h"
        exp = DatetimeIndex(
            ["2011-01-01 07:00", "2011-01-01 08:00", "2011-01-01 09:00"],
            name="xxx",
            tz=tz,
            freq="h",
        ).as_unit(unit)
        # 断言 idx 同于 idx 自身在 freq="h" 下移动 -3 个单位后的结果为 exp
        tm.assert_index_equal(idx.shift(-3, freq="h"), exp)

    # 定义测试方法 test_dti_shift_freqs，使用 unit 作为参数
    def test_dti_shift_freqs(self, unit):
        # test shift for DatetimeIndex and non DatetimeIndex
        # GH#8083
        # 创建一个时间范围对象 drange，从 "20130101" 开始，周期为 5，单位为 unit
        drange = date_range("20130101", periods=5, unit=unit)
        # 对 drange 进行向后移动 1 个单位
        result = drange.shift(1)
        # 创建预期的 DatetimeIndex 对象 expected
        expected = DatetimeIndex(
            ["2013-01-02", "2013-01-03", "2013-01-04", "2013-01-05", "2013-01-06"],
            dtype=f"M8[{unit}]",
            freq="D",
        )
        # 断言 result 同于 expected
        tm.assert_index_equal(result, expected)

        # 对 drange 进行向前移动 1 个单位
        result = drange.shift(-1)
        # 创建预期的 DatetimeIndex 对象 expected
        expected = DatetimeIndex(
            ["2012-12-31", "2013-01-01", "2013-01-02", "2013-01-03", "2013-01-04"],
            dtype=f"M8[{unit}]",
            freq="D",
        )
        # 断言 result 同于 expected
        tm.assert_index_equal(result, expected)

        # 对 drange 进行向后移动 3 个单位，频率为 "2D"
        result = drange.shift(3, freq="2D")
        # 创建预期的 DatetimeIndex 对象 expected
        expected = DatetimeIndex(
            ["2013-01-07", "2013-01-08", "2013-01-09", "2013-01-10", "2013-01-11"],
            dtype=f"M8[{unit}]",
            freq="D",
        )
        # 断言 result 同于 expected
        tm.assert_index_equal(result, expected)

    # 定义测试方法 test_dti_shift_int，使用 unit 作为参数
    def test_dti_shift_int(self, unit):
        # 创建一个时间范围对象 rng，从 "1/1/2000" 开始，周期为 20，单位为 unit
        rng = date_range("1/1/2000", periods=20, unit=unit)

        # 将 rng 中的每个时间戳向后移动 5 个单位
        result = rng + 5 * rng.freq
        # 创建预期的 DatetimeIndex 对象 expected
        expected = rng.shift(5)
        # 断言 result 同于 expected
        tm.assert_index_equal(result, expected)

        # 将 rng 中的每个时间戳向前移动 5 个单位
        result = rng - 5 * rng.freq
        # 创建预期的 DatetimeIndex 对象 expected
        expected = rng.shift(-5)
        # 断言 result 同于 expected
        tm.assert_index_equal(result, expected)
    # 测试在没有频率情况下进行日期时间索引的偏移
    def test_dti_shift_no_freq(self, unit):
        # GH#19147
        # 创建一个没有频率的日期时间索引
        dti = DatetimeIndex(["2011-01-01 10:00", "2011-01-01"], freq=None).as_unit(unit)
        # 使用 pytest 检查是否会引发 NullFrequencyError 异常，并匹配特定错误信息
        with pytest.raises(NullFrequencyError, match="Cannot shift with no freq"):
            # 进行偏移操作
            dti.shift(2)

    # 测试本地化日期时间索引的偏移
    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_shift_localized(self, tzstr, unit):
        # 创建一个带有频率的日期范围对象
        dr = date_range("2011/1/1", "2012/1/1", freq="W-FRI", unit=unit)
        # 将日期时间索引对象本地化
        dr_tz = dr.tz_localize(tzstr)

        # 进行偏移操作，将索引移动 1 个单位，每个单位为 10 分钟
        result = dr_tz.shift(1, "10min")
        # 断言偏移后的结果保留了原始时区信息
        assert result.tz == dr_tz.tz

    # 测试跨夏令时的日期时间索引偏移
    def test_dti_shift_across_dst(self, unit):
        # GH 8616
        # 创建一个带有时区信息的日期范围对象
        idx = date_range(
            "2013-11-03", tz="America/Chicago", periods=7, freq="h", unit=unit
        )
        # 创建一个数据序列，索引去掉最后一个时间戳
        ser = Series(index=idx[:-1], dtype=object)
        # 进行频率为小时的偏移操作
        result = ser.shift(freq="h")
        # 创建一个预期的数据序列，索引去掉第一个时间戳
        expected = Series(index=idx[1:], dtype=object)
        # 断言偏移后的结果与预期结果相等
        tm.assert_series_equal(result, expected)

    # 测试接近午夜时的日期时间索引偏移
    @pytest.mark.parametrize(
        "shift, result_time",
        [
            [0, "2014-11-14 00:00:00"],
            [-1, "2014-11-13 23:00:00"],
            [1, "2014-11-14 01:00:00"],
        ],
    )
    def test_dti_shift_near_midnight(self, shift, result_time, unit):
        # GH 8616
        # 创建一个带有时区信息的日期时间对象
        tz = zoneinfo.ZoneInfo("US/Eastern")
        dt_est = datetime(2014, 11, 14, 0, tzinfo=tz)
        # 创建一个日期时间索引对象
        idx = DatetimeIndex([dt_est]).as_unit(unit)
        # 创建一个数据序列，包含单个数据值
        ser = Series(data=[1], index=idx)
        # 进行频率为小时的偏移操作
        result = ser.shift(shift, freq="h")
        # 创建一个预期的数据序列，索引包含预期的时间戳
        exp_index = DatetimeIndex([result_time], tz=tz).as_unit(unit)
        expected = Series(1, index=exp_index)
        # 断言偏移后的结果与预期结果相等
        tm.assert_series_equal(result, expected)

    # 测试日期时间索引的周期性偏移
    def test_shift_periods(self, unit):
        # GH#22458 : argument 'n' was deprecated in favor of 'periods'
        # 创建一个带有指定周期数的日期范围对象
        idx = date_range(
            start=datetime(2009, 1, 1), end=datetime(2010, 1, 1), periods=3, unit=unit
        )
        # 断言索引偏移 0 个周期后结果与原始索引相等
        tm.assert_index_equal(idx.shift(periods=0), idx)
        # 断言索引偏移 0 个周期后结果与原始索引相等（使用 'shift' 参数）
        tm.assert_index_equal(idx.shift(0), idx)

    # 测试工作日偏移
    @pytest.mark.parametrize("freq", ["B", "C"])
    def test_shift_bday(self, freq, unit):
        # 创建一个带有指定频率的日期范围对象
        rng = date_range(
            datetime(2009, 1, 1), datetime(2010, 1, 1), freq=freq, unit=unit
        )
        # 进行偏移操作，将索引向后移动 5 个单位
        shifted = rng.shift(5)
        # 断言偏移后的结果的第一个时间戳与原始索引的第 5 个时间戳相等
        assert shifted[0] == rng[5]
        # 断言偏移后的结果的频率与原始索引的频率相等
        assert shifted.freq == rng.freq

        # 进行偏移操作，将索引向前移动 5 个单位
        shifted = rng.shift(-5)
        # 断言偏移后的结果的第 6 个时间戳与原始索引的第一个时间戳相等
        assert shifted[5] == rng[0]
        # 断言偏移后的结果的频率与原始索引的频率相等
        assert shifted.freq == rng.freq

        # 进行偏移操作，不移动索引
        shifted = rng.shift(0)
        # 断言偏移后的结果与原始索引相等
        assert shifted[0] == rng[0]
        # 断言偏移后的结果的频率与原始索引的频率相等
        assert shifted.freq == rng.freq
    # 定义一个测试函数，测试日期范围的偏移计算
    def test_shift_bmonth(self, performance_warning, unit):
        # 创建一个日期范围，从 2009 年 1 月 1 日到 2010 年 1 月 1 日，频率为每月最后一个工作日，单位由参数指定
        rng = date_range(
            datetime(2009, 1, 1),
            datetime(2010, 1, 1),
            freq=pd.offsets.BMonthEnd(),
            unit=unit,
        )
        # 对日期范围进行偏移，向后偏移一天的频率
        shifted = rng.shift(1, freq=pd.offsets.BDay())
        # 断言偏移后的第一个日期是否等于原始范围的第一个日期加上一个工作日的偏移量
        assert shifted[0] == rng[0] + pd.offsets.BDay()

        # 再次创建相同的日期范围，用不同的频率参数
        rng = date_range(
            datetime(2009, 1, 1),
            datetime(2010, 1, 1),
            freq=pd.offsets.BMonthEnd(),
            unit=unit,
        )
        # 使用断言来验证在性能警告下进行偏移计算
        with tm.assert_produces_warning(performance_warning):
            # 对日期范围进行偏移，使用自定义频率 pd.offsets.CDay()
            shifted = rng.shift(1, freq=pd.offsets.CDay())
            # 断言偏移后的第一个日期是否等于原始范围的第一个日期加上一个自定义频率的偏移量
            assert shifted[0] == rng[0] + pd.offsets.CDay()

    # 定义另一个测试函数，测试空的日期范围偏移情况
    def test_shift_empty(self, unit):
        # GH#14811：测试空的日期范围的偏移行为
        dti = date_range(start="2016-10-21", end="2016-10-21", freq="BME", unit=unit)
        # 对空日期范围进行偏移，偏移量为1
        result = dti.shift(1)
        # 使用断言验证偏移后的结果是否与原始日期范围相等
        tm.assert_index_equal(result, dti)
```