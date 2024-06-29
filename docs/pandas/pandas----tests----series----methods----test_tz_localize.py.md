# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_tz_localize.py`

```
# 从 datetime 模块导入 timezone 类
from datetime import timezone

# 导入 pytest 测试框架
import pytest
# 导入 pytz 用于处理时区
import pytz

# 从 pandas._libs.tslibs 中导入 timezones
from pandas._libs.tslibs import timezones

# 从 pandas 中导入以下对象：
# DatetimeIndex 用于处理日期时间索引
# NaT 代表缺失的日期时间值
# Series 用于处理序列数据
# Timestamp 用于表示单个时间戳
# date_range 用于生成日期时间范围
from pandas import (
    DatetimeIndex,
    NaT,
    Series,
    Timestamp,
    date_range,
)
# 导入 pandas 测试工具模块
import pandas._testing as tm

# 定义 TestTZLocalize 测试类
class TestTZLocalize:
    # 定义 test_series_tz_localize_ambiguous_bool 方法
    def test_series_tz_localize_ambiguous_bool(self):
        # 确保能正确接受布尔值作为模糊时间的处理方式
        
        # 创建 Timestamp 对象 ts，表示时间为 "2015-11-01 01:00:03"
        ts = Timestamp("2015-11-01 01:00:03")
        # 创建带时区信息的 Timestamp 对象 expected0，表示时间为 "2015-11-01 01:00:03"，时区为 "US/Central"
        expected0 = Timestamp("2015-11-01 01:00:03-0500", tz="US/Central")
        # 创建带时区信息的 Timestamp 对象 expected1，表示时间为 "2015-11-01 01:00:03"，时区为 "US/Central"
        expected1 = Timestamp("2015-11-01 01:00:03-0600", tz="US/Central")

        # 创建 Series 对象 ser，包含单个时间戳 ts
        ser = Series([ts])
        # 创建带时区信息的 Series 对象 expected0，包含单个时间戳 expected0
        expected0 = Series([expected0])
        # 创建带时区信息的 Series 对象 expected1，包含单个时间戳 expected1

        # 使用 pytest 的 external_error_raised 上下文，捕获 pytz.AmbiguousTimeError 异常
        with tm.external_error_raised(pytz.AmbiguousTimeError):
            # 对 ser 的日期时间进行时区本地化为 "US/Central"
            ser.dt.tz_localize("US/Central")

        # 将 ser 的日期时间以 ambiguous=True 的方式进行时区本地化为 "US/Central"，并将结果与 expected0 对比
        result = ser.dt.tz_localize("US/Central", ambiguous=True)
        tm.assert_series_equal(result, expected0)

        # 将 ser 的日期时间以 ambiguous=[True] 的方式进行时区本地化为 "US/Central"，并将结果与 expected0 对比
        result = ser.dt.tz_localize("US/Central", ambiguous=[True])
        tm.assert_series_equal(result, expected0)

        # 将 ser 的日期时间以 ambiguous=False 的方式进行时区本地化为 "US/Central"，并将结果与 expected1 对比
        result = ser.dt.tz_localize("US/Central", ambiguous=False)
        tm.assert_series_equal(result, expected1)

        # 将 ser 的日期时间以 ambiguous=[False] 的方式进行时区本地化为 "US/Central"，并将结果与 expected1 对比
        result = ser.dt.tz_localize("US/Central", ambiguous=[False])
        tm.assert_series_equal(result, expected1)

    # 定义 test_series_tz_localize_matching_index 方法
    def test_series_tz_localize_matching_index(self):
        # 确保结果的索引与原始序列的索引匹配
        # GH 43080

        # 创建 Series 对象 dt_series，包含从 "2021-01-01T02:00:00" 开始的五个日期时间，频率为每天一次
        # 索引为 [2, 6, 7, 8, 11]，数据类型为 "category"
        dt_series = Series(
            date_range(start="2021-01-01T02:00:00", periods=5, freq="1D"),
            index=[2, 6, 7, 8, 11],
            dtype="category",
        )

        # 对 dt_series 的日期时间进行时区本地化为 "Europe/Berlin"
        result = dt_series.dt.tz_localize("Europe/Berlin")

        # 创建带时区信息的 Series 对象 expected，包含从 "2021-01-01T02:00:00" 开始的五个日期时间，频率为每天一次，时区为 "Europe/Berlin"，索引与 dt_series 相同
        expected = Series(
            date_range(
                start="2021-01-01T02:00:00", periods=5, freq="1D", tz="Europe/Berlin"
            ),
            index=[2, 6, 7, 8, 11],
        )

        # 检查 result 与 expected 是否相等
        tm.assert_series_equal(result, expected)

    # 使用 pytest 的 parametrize 装饰器，定义多个参数化测试
    @pytest.mark.parametrize(
        "method, exp",
        [
            ["shift_forward", "2015-03-29 03:00:00"],
            ["shift_backward", "2015-03-29 01:59:59.999999999"],
            ["NaT", NaT],
            ["raise", None],
            ["foo", "invalid"],
        ],
    )
    # 定义测试方法，用于测试时区本地化对于不存在时间的处理
    def test_tz_localize_nonexistent(self, warsaw, method, exp, unit):
        # GH 8917
        # 设定时区为华沙时区
        tz = warsaw
        # 定义生成日期范围的起始时间、周期数、频率和单位
        n = 60
        dti = date_range(start="2015-03-29 02:00:00", periods=n, freq="min", unit=unit)
        # 创建一个带索引的 Series 对象
        ser = Series(1, index=dti)
        # 将 Series 转换为 DataFrame
        df = ser.to_frame()

        # 根据不同的处理方法进行测试
        if method == "raise":
            # 检查在引发异常时是否会捕获 pytz.NonExistentTimeError
            with tm.external_error_raised(pytz.NonExistentTimeError):
                dti.tz_localize(tz, nonexistent=method)
            with tm.external_error_raised(pytz.NonExistentTimeError):
                ser.tz_localize(tz, nonexistent=method)
            with tm.external_error_raised(pytz.NonExistentTimeError):
                df.tz_localize(tz, nonexistent=method)

        elif exp == "invalid":
            # 如果 exp 为 "invalid"，则检查是否会引发 ValueError 异常
            msg = (
                "The nonexistent argument must be one of "
                "'raise', 'NaT', 'shift_forward', 'shift_backward' "
                "or a timedelta object"
            )
            with pytest.raises(ValueError, match=msg):
                dti.tz_localize(tz, nonexistent=method)
            with pytest.raises(ValueError, match=msg):
                ser.tz_localize(tz, nonexistent=method)
            with pytest.raises(ValueError, match=msg):
                df.tz_localize(tz, nonexistent=method)

        else:
            # 否则，进行预期结果的比较
            # 对 Series 进行时区本地化，并比较结果与预期结果的相等性
            result = ser.tz_localize(tz, nonexistent=method)
            expected = Series(1, index=DatetimeIndex([exp] * n, tz=tz).as_unit(unit))
            tm.assert_series_equal(result, expected)

            # 对 DataFrame 进行时区本地化，并比较结果与预期结果的相等性
            result = df.tz_localize(tz, nonexistent=method)
            expected = expected.to_frame()
            tm.assert_frame_equal(result, expected)

            # 对日期范围的索引进行时区本地化，并比较结果与预期索引的相等性
            res_index = dti.tz_localize(tz, nonexistent=method)
            tm.assert_index_equal(res_index, expected.index)

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    # 定义测试方法，用于测试空 Series 对象的时区本地化
    def test_series_tz_localize_empty(self, tzstr):
        # GH#2248
        # 创建一个数据类型为对象的空 Series 对象
        ser = Series(dtype=object)

        # 将空 Series 对象本地化到 UTC 时区，并验证索引是否为 UTC 时区
        ser2 = ser.tz_localize("utc")
        assert ser2.index.tz == timezone.utc

        # 将空 Series 对象本地化到指定时区 tzstr，并验证索引的时区
        ser2 = ser.tz_localize(tzstr)
        timezones.tz_compare(ser2.index.tz, timezones.maybe_get_tz(tzstr))
```