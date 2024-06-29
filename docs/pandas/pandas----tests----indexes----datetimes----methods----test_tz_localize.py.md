# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_tz_localize.py`

```
from datetime import (
    datetime,           # 导入 datetime 类
    timedelta,          # 导入 timedelta 类
    timezone,           # 导入 timezone 类
)
from zoneinfo import ZoneInfo  # 导入 ZoneInfo 类

import dateutil.tz       # 导入 dateutil.tz 模块
from dateutil.tz import gettz   # 导入 gettz 函数
import numpy as np       # 导入 NumPy 库
import pytest            # 导入 pytest 库
import pytz              # 导入 pytz 库

from pandas import (
    DatetimeIndex,      # 导入 DatetimeIndex 类
    Timestamp,          # 导入 Timestamp 类
    bdate_range,        # 导入 bdate_range 函数
    date_range,         # 导入 date_range 函数
    offsets,            # 导入 offsets 模块
    to_datetime,        # 导入 to_datetime 函数
)
import pandas._testing as tm  # 导入 pandas._testing 模块作为 tm 别名


@pytest.fixture(params=["pytz/US/Eastern", gettz("US/Eastern"), ZoneInfo("US/Eastern")])
def tz(request):
    if isinstance(request.param, str) and request.param.startswith("pytz/"):
        pytz = pytest.importorskip("pytz")  # 如果参数是以 "pytz/" 开头的字符串，导入并返回对应时区对象
        return pytz.timezone(request.param.removeprefix("pytz/"))
    return request.param  # 返回参数本身


class TestTZLocalize:
    def test_tz_localize_invalidates_freq(self):
        # 只在非歧义情况下保留频率

        # 如果本地化为 US/Eastern，这会跨越夏令时转换
        dti = date_range("2014-03-08 23:00", "2014-03-09 09:00", freq="h")
        assert dti.freq == "h"  # 断言频率为 "h"

        result = dti.tz_localize(None)  # 不进行操作
        assert result.freq == "h"  # 断言结果的频率为 "h"

        result = dti.tz_localize("UTC")  # 保留非歧义的频率
        assert result.freq == "h"  # 断言结果的频率为 "h"

        result = dti.tz_localize("US/Eastern", nonexistent="shift_forward")
        assert result.freq is None  # 断言结果的频率为 None
        assert result.inferred_freq is None  # 断言推断频率为 None，即我们不会过于严格

        # 可以保留频率的情况，因为长度为 1
        dti2 = dti[:1]
        result = dti2.tz_localize("US/Eastern")
        assert result.freq == "h"  # 断言结果的频率为 "h"

    def test_tz_localize_utc_copies(self, utc_fixture):
        # GH#46460
        times = ["2015-03-08 01:00", "2015-03-08 02:00", "2015-03-08 03:00"]
        index = DatetimeIndex(times)

        res = index.tz_localize(utc_fixture)
        assert not tm.shares_memory(res, index)  # 断言结果对象与原始对象不共享内存

        res2 = index._data.tz_localize(utc_fixture)
        assert not tm.shares_memory(index._data, res2)  # 断言结果数据与原始数据不共享内存

    def test_dti_tz_localize_nonexistent_raise_coerce(self):
        # GH#13057
        times = ["2015-03-08 01:00", "2015-03-08 02:00", "2015-03-08 03:00"]
        index = DatetimeIndex(times)
        tz = "US/Eastern"

        # 使用 pytest 断言检查是否引发了 pytz.NonExistentTimeError 异常，且异常消息包含 times 中任一时间字符串
        with pytest.raises(pytz.NonExistentTimeError, match="|".join(times)):
            index.tz_localize(tz=tz)

        with pytest.raises(pytz.NonExistentTimeError, match="|".join(times)):
            index.tz_localize(tz=tz, nonexistent="raise")

        result = index.tz_localize(tz=tz, nonexistent="NaT")
        
        # 创建预期的时间序列对象，将其转换为 US/Eastern 时区，并使用 pandas._testing 模块的断言函数比较结果
        test_times = ["2015-03-08 01:00-05:00", "NaT", "2015-03-08 03:00-04:00"]
        dti = to_datetime(test_times, utc=True)
        expected = dti.tz_convert("US/Eastern")
        tm.assert_index_equal(result, expected)
    # 测试函数：测试在时区本地化时的模糊时间推断功能
    def test_dti_tz_localize_ambiguous_infer(self, tz):
        # 创建一个日期范围，从2011年11月6日开始，每小时一个时间点，共5个时间点
        dr = date_range(datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour())
        # 使用 pytest 检查是否会引发 pytz 的 AmbiguousTimeError 异常，异常信息中包含"Cannot infer dst time"
        with pytest.raises(pytz.AmbiguousTimeError, match="Cannot infer dst time"):
            # 尝试在指定时区下对日期范围进行本地化
            dr.tz_localize(tz)

    # 测试函数：测试在时区本地化时的模糊时间推断功能（第二个测试用例）
    def test_dti_tz_localize_ambiguous_infer2(self, tz, unit):
        # 创建一个日期范围，从2011年11月6日开始，每小时一个时间点，共5个时间点，并指定时区和单位
        dr = date_range(
            datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour(), tz=tz, unit=unit
        )
        # 创建一个时间列表，包含重复的小时（模拟夏令时转变情况）
        times = [
            "11/06/2011 00:00",
            "11/06/2011 01:00",
            "11/06/2011 01:00",
            "11/06/2011 02:00",
            "11/06/2011 03:00",
        ]
        # 创建 DatetimeIndex 对象，并按单位调整
        di = DatetimeIndex(times).as_unit(unit)
        # 在时区本地化时，使用 "infer" 模式推断模糊时间
        result = di.tz_localize(tz, ambiguous="infer")
        # 期望结果是在不受频率限制的日期范围中
        expected = dr._with_freq(None)
        # 使用 assert_index_equal 检查结果与期望是否一致
        tm.assert_index_equal(result, expected)
        # 创建另一个 DatetimeIndex 对象，并按单位调整
        result2 = DatetimeIndex(times, tz=tz, ambiguous="infer").as_unit(unit)
        # 使用 assert_index_equal 检查结果与期望是否一致
        tm.assert_index_equal(result2, expected)

    # 测试函数：测试在时区本地化时的模糊时间推断功能（第三个测试用例）
    def test_dti_tz_localize_ambiguous_infer3(self, tz):
        # 创建一个日期范围，从2011年6月1日开始，每小时一个时间点，共10个时间点
        dr = date_range(datetime(2011, 6, 1, 0), periods=10, freq=offsets.Hour())
        # 对日期范围进行时区本地化
        localized = dr.tz_localize(tz)
        # 在时区本地化时，使用 "infer" 模式推断模糊时间
        localized_infer = dr.tz_localize(tz, ambiguous="infer")
        # 使用 assert_index_equal 检查两个本地化结果是否一致
        tm.assert_index_equal(localized, localized_infer)

    # 测试函数：测试在时区本地化时的非法时间处理
    def test_dti_tz_localize_ambiguous_times(self, tz):
        # 创建一个日期范围，从2011年3月13日凌晨1点30分开始，每小时一个时间点，共3个时间点
        dr = date_range(datetime(2011, 3, 13, 1, 30), periods=3, freq=offsets.Hour())
        # 使用 pytest 检查是否会引发 pytz 的 NonExistentTimeError 异常，异常信息中包含"2011-03-13 02:30:00"
        with pytest.raises(pytz.NonExistentTimeError, match="2011-03-13 02:30:00"):
            # 尝试在指定时区下对日期范围进行本地化
            dr.tz_localize(tz)

        # 创建一个日期范围，从2011年3月13日凌晨3点30分开始，每小时一个时间点，共3个时间点，并指定时区
        dr = date_range(
            datetime(2011, 3, 13, 3, 30), periods=3, freq=offsets.Hour(), tz=tz
        )

        # 创建一个日期范围，从2011年11月6日凌晨1点30分开始，每小时一个时间点，共3个时间点
        dr = date_range(datetime(2011, 11, 6, 1, 30), periods=3, freq=offsets.Hour())
        # 使用 pytest 检查是否会引发 pytz 的 AmbiguousTimeError 异常，异常信息中包含"Cannot infer dst time"
        with pytest.raises(pytz.AmbiguousTimeError, match="Cannot infer dst time"):
            # 尝试在指定时区下对日期范围进行本地化
            dr.tz_localize(tz)

        # 创建一个日期范围，从2011年3月13日开始，每30分钟一个时间点，共48个时间点，并指定为 UTC 时区
        dr = date_range(
            datetime(2011, 3, 13), periods=48, freq=offsets.Minute(30), tz=timezone.utc
        )

    # 测试函数：测试将日期时间索引从本地时区转换为 UTC 时区
    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_tz_localize_pass_dates_to_utc(self, tzstr):
        # 创建字符串日期列表
        strdates = ["1/1/2012", "3/1/2012", "4/1/2012"]
        # 创建 DatetimeIndex 对象
        idx = DatetimeIndex(strdates)
        # 将日期时间索引从字符串日期列表中的时区字符串进行本地化
        conv = idx.tz_localize(tzstr)
        # 使用时区字符串创建另一个 DatetimeIndex 对象
        fromdates = DatetimeIndex(strdates, tz=tzstr)
        # 使用 assert 检查两个本地化结果的时区是否相同
        assert conv.tz == fromdates.tz
        # 使用 assert 检查两个本地化结果的值是否相同
        tm.assert_numpy_array_equal(conv.values, fromdates.values)

    # 测试函数：使用参数化测试不同的时区字符串
    @pytest.mark.parametrize("prefix", ["", "dateutil/"])
    # 测试时区本地化功能，并使用指定的前缀生成目标时区字符串
    def test_dti_tz_localize(self, prefix):
        # 组合前缀和时区字符串生成完整的时区名称
        tzstr = prefix + "US/Eastern"
        # 创建一个包含毫秒级频率时间索引，从 "1/1/2005" 到 "1/1/2005 0:00:30.256"
        dti = date_range(start="1/1/2005", end="1/1/2005 0:00:30.256", freq="ms")
        # 将时间索引本地化为指定时区
        dti2 = dti.tz_localize(tzstr)

        # 创建一个包含时区为 UTC 的时间索引，从 "1/1/2005 05:00" 到 "1/1/2005 5:00:30.256"
        dti_utc = date_range(
            start="1/1/2005 05:00", end="1/1/2005 5:00:30.256", freq="ms", tz="utc"
        )

        # 检查本地化后的时间索引与 UTC 时间索引的数值是否相等
        tm.assert_numpy_array_equal(dti2.values, dti_utc.values)

        # 将已本地化的时间索引转换为另一个时区
        dti3 = dti2.tz_convert(prefix + "US/Pacific")
        # 检查转换后的时间索引与 UTC 时间索引的数值是否相等
        tm.assert_numpy_array_equal(dti3.values, dti_utc.values)

        # 创建一个包含夏令时切换时出现歧义的时间索引，从 "11/6/2011 1:59" 到 "11/6/2011 2:00"
        dti = date_range(start="11/6/2011 1:59", end="11/6/2011 2:00", freq="ms")
        # 断言本地化时会引发 AmbiguousTimeError 异常，匹配指定的错误消息
        with pytest.raises(pytz.AmbiguousTimeError, match="Cannot infer dst time"):
            dti.tz_localize(tzstr)

        # 创建一个包含不存在的时间点时引发异常的时间索引，从 "3/13/2011 1:59" 到 "3/13/2011 2:00"
        dti = date_range(start="3/13/2011 1:59", end="3/13/2011 2:00", freq="ms")
        # 断言本地化时会引发 NonExistentTimeError 异常，匹配指定的错误消息
        with pytest.raises(pytz.NonExistentTimeError, match="2011-03-13 02:00:00"):
            dti.tz_localize(tzstr)

    # 测试时间索引本地化并进行 UTC 转换的功能
    def test_dti_tz_localize_utc_conversion(self, tz):
        # 本地化到时区应当：
        #  1) 检查夏令时的歧义性
        #  2) 转换为 UTC

        # 创建一个时间范围对象，从 "3/10/2012" 到 "3/11/2012"，频率为每 30 分钟
        rng = date_range("3/10/2012", "3/11/2012", freq="30min")

        # 将时间范围对象本地化到指定时区
        converted = rng.tz_localize(tz)
        # 生成预期的本地化后的无时区时间范围对象
        expected_naive = rng + offsets.Hour(5)
        # 断言本地化后的时间戳数组与预期的无时区时间戳数组是否相等
        tm.assert_numpy_array_equal(converted.asi8, expected_naive.asi8)

        # 创建一个包含夏令时切换时出现歧义的时间范围对象，从 "3/11/2012" 到 "3/12/2012"，频率为每 30 分钟
        rng = date_range("3/11/2012", "3/12/2012", freq="30min")
        # 断言本地化时会引发 NonExistentTimeError 异常，匹配指定的错误消息
        with pytest.raises(pytz.NonExistentTimeError, match="2012-03-11 02:00:00"):
            rng.tz_localize(tz)

    # 测试时间索引本地化并进行逆操作的功能
    def test_dti_tz_localize_roundtrip(self, tz_aware_fixture):
        # 注意：此测试用例验证了当范围内没有夏令时转换时，可以成功地对无时区时间索引进行本地化和逆本地化操作。
        # 创建一个时间索引对象，从 "2014-06-01" 到 "2014-08-30"，频率为每 15 分钟
        idx = date_range(start="2014-06-01", end="2014-08-30", freq="15min")
        # 获取时区对象
        tz = tz_aware_fixture
        # 将时间索引本地化到指定时区
        localized = idx.tz_localize(tz)
        # 断言本地化时会引发 TypeError 异常，匹配指定的错误消息
        with pytest.raises(
            TypeError, match="Already tz-aware, use tz_convert to convert"
        ):
            localized.tz_localize(tz)
        # 将已本地化的时间索引逆本地化（移除时区信息）
        reset = localized.tz_localize(None)
        # 断言重置后的时间索引没有时区信息
        assert reset.tzinfo is None
        # 生成预期的无时区时间索引
        expected = idx._with_freq(None)
        # 断言重置后的时间索引与预期的无时区时间索引相等
        tm.assert_index_equal(reset, expected)

    # 测试将无时区时间索引本地化到具体时区的功能
    def test_dti_tz_localize_naive(self):
        # 创建一个时间索引对象，从 "1/1/2011" 开始，包含 100 个周期，频率为每小时
        rng = date_range("1/1/2011", periods=100, freq="h")

        # 将时间索引本地化到指定时区 "US/Pacific"
        conv = rng.tz_localize("US/Pacific")
        # 生成预期的带时区信息的时间索引对象
        exp = date_range("1/1/2011", periods=100, freq="h", tz="US/Pacific")

        # 断言本地化后的时间索引与预期的时间索引相等（不含时区信息）
        tm.assert_index_equal(conv, exp._with_freq(None))
    def test_dti_tz_localize_tzlocal(self):
        # 测试函数：test_dti_tz_localize_tzlocal
        # 测试日期时间处理模块中的时区本地化功能
        # GH#13583：GitHub 上的 issue 编号 13583

        # 获取本地时区对于指定日期时间的偏移量，并转换为纳秒单位
        offset = dateutil.tz.tzlocal().utcoffset(datetime(2011, 1, 1))
        offset = int(offset.total_seconds() * 1000000000)

        # 创建一个从 "2001-01-01" 到 "2001-03-01" 的日期范围对象
        dti = date_range(start="2001-01-01", end="2001-03-01")

        # 将日期时间范围对象本地化到本地时区
        dti2 = dti.tz_localize(dateutil.tz.tzlocal())

        # 断言两个 numpy 数组是否相等，验证本地化后的日期时间的纳秒表示加上偏移量是否等于原始日期时间的纳秒表示
        tm.assert_numpy_array_equal(dti2.asi8 + offset, dti.asi8)

        # 创建一个从 "2001-01-01" 到 "2001-03-01" 的日期范围对象，并直接在创建时本地化到本地时区
        dti = date_range(start="2001-01-01", end="2001-03-01", tz=dateutil.tz.tzlocal())

        # 将本地化后的日期时间范围对象移除时区信息
        dti2 = dti.tz_localize(None)

        # 断言两个 numpy 数组是否相等，验证移除时区信息后的日期时间的纳秒表示减去偏移量是否等于原始日期时间的纳秒表示
        tm.assert_numpy_array_equal(dti2.asi8 - offset, dti.asi8)


    def test_dti_tz_localize_ambiguous_nat(self, tz):
        # 测试函数：test_dti_tz_localize_ambiguous_nat
        # 测试日期时间处理模块中的时区本地化功能，处理模糊时间的情况，使用 NaT 表示不明确的时间

        # 定义一组时间字符串列表
        times = [
            "11/06/2011 00:00",
            "11/06/2011 01:00",
            "11/06/2011 01:00",
            "11/06/2011 02:00",
            "11/06/2011 03:00",
        ]

        # 创建日期时间索引对象
        di = DatetimeIndex(times)

        # 将日期时间索引对象本地化到指定的时区，对于模糊时间使用 NaT 表示
        localized = di.tz_localize(tz, ambiguous="NaT")

        # 定义另一组时间字符串列表
        times = [
            "11/06/2011 00:00",
            np.nan,
            np.nan,
            "11/06/2011 02:00",
            "11/06/2011 03:00",
        ]

        # 创建具有时区信息的日期时间索引对象
        di_test = DatetimeIndex(times, tz="US/Eastern")

        # 断言两个 numpy 数组是否相等，验证具有时区信息和模糊时间处理的日期时间索引对象是否一致
        # 左侧数据类型为 datetime64[ns, US/Eastern]
        # 右侧数据类型为 datetime64[ns, tzfile('/usr/share/zoneinfo/US/Eastern')]
        tm.assert_numpy_array_equal(di_test.values, localized.values)
    def test_dti_tz_localize_ambiguous_flags(self, tz, unit):
        # November 6, 2011, fall back, repeat 2 AM hour
        # 2011年11月6日，夏令时结束，2点钟再次重复

        # Pass in flags to determine right dst transition
        # 传入标志位以确定正确的夏令时转换
        dr = date_range(
            datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour(), tz=tz, unit=unit
        )
        # 创建一个日期范围，从2011年11月6日00:00开始，每小时生成一个日期时间，带有时区和单位

        times = [
            "11/06/2011 00:00",
            "11/06/2011 01:00",
            "11/06/2011 01:00",
            "11/06/2011 02:00",
            "11/06/2011 03:00",
        ]

        # Test tz_localize
        # 测试 tz_localize 方法
        di = DatetimeIndex(times).as_unit(unit)
        # 将时间列表转换为 DateTimeIndex，并指定单位

        is_dst = [1, 1, 0, 0, 0]
        # 夏令时标志位数组，表示每个时间是否处于夏令时

        localized = di.tz_localize(tz, ambiguous=is_dst)
        # 使用时区进行本地化，并传入夏令时标志位
        expected = dr._with_freq(None)
        # 期望的结果，具有相同的频率

        tm.assert_index_equal(expected, localized)
        # 使用测试框架验证预期结果与本地化后的结果是否相等

        result = DatetimeIndex(times, tz=tz, ambiguous=is_dst).as_unit(unit)
        # 创建一个带有时区和夏令时标志位的 DateTimeIndex，并指定单位

        tm.assert_index_equal(result, expected)
        # 使用测试框架验证结果与预期结果是否相等

        localized = di.tz_localize(tz, ambiguous=np.array(is_dst))
        # 使用 numpy 数组作为夏令时标志位进行本地化

        tm.assert_index_equal(dr, localized)
        # 使用测试框架验证期望的日期范围与本地化后的结果是否相等

        localized = di.tz_localize(tz, ambiguous=np.array(is_dst).astype("bool"))
        # 使用布尔类型的 numpy 数组作为夏令时标志位进行本地化

        # Test constructor
        # 测试构造函数

        localized = DatetimeIndex(times, tz=tz, ambiguous=is_dst).as_unit(unit)
        # 使用构造函数创建 DateTimeIndex，并指定时区和夏令时标志位

        tm.assert_index_equal(dr, localized)
        # 使用测试框架验证期望的日期范围与本地化后的结果是否相等

        # Test duplicate times where inferring the dst fails
        # 测试重复时间的情况，推断夏令时失败

        times += times
        # 时间列表加倍

        di = DatetimeIndex(times).as_unit(unit)
        # 将时间列表转换为 DateTimeIndex，并指定单位

        # When the sizes are incompatible, make sure error is raised
        # 当大小不兼容时，确保引发错误
        msg = "Length of ambiguous bool-array must be the same size as vals"
        # 错误信息

        with pytest.raises(Exception, match=msg):
            # 使用 pytest 断言应引发异常，并匹配特定的错误消息
            di.tz_localize(tz, ambiguous=is_dst)

        # When sizes are compatible and there are repeats ('infer' won't work)
        # 当大小兼容且存在重复时（'infer' 将无法工作）
        is_dst = np.hstack((is_dst, is_dst))
        # 将夏令时标志位数组扩展为两倍长度

        localized = di.tz_localize(tz, ambiguous=is_dst)
        # 使用扩展后的夏令时标志位进行本地化

        dr = dr.append(dr)
        # 将日期范围对象追加一次，使其长度加倍

        tm.assert_index_equal(dr, localized)
        # 使用测试框架验证期望的日期范围与本地化后的结果是否相等

    def test_dti_tz_localize_ambiguous_flags2(self, tz):
        # When there is no dst transition, nothing special happens
        # 当没有夏令时转换时，不会发生特殊情况

        dr = date_range(datetime(2011, 6, 1, 0), periods=10, freq=offsets.Hour())
        # 创建一个日期范围，从2011年6月1日00:00开始，每小时生成一个日期时间

        is_dst = np.array([1] * 10)
        # 夏令时标志位数组，表示所有时间都处于夏令时

        localized = dr.tz_localize(tz)
        # 使用时区进行本地化

        localized_is_dst = dr.tz_localize(tz, ambiguous=is_dst)
        # 使用夏令时标志位进行本地化

        tm.assert_index_equal(localized, localized_is_dst)
        # 使用测试框架验证两次本地化的结果是否相等

    def test_dti_tz_localize_bdate_range(self):
        dr = bdate_range("1/1/2009", "1/1/2010")
        # 创建一个商业日期范围，从2009年1月1日到2010年1月1日

        dr_utc = bdate_range("1/1/2009", "1/1/2010", tz=timezone.utc)
        # 创建一个使用 UTC 时区的商业日期范围，从2009年1月1日到2010年1月1日

        localized = dr.tz_localize(timezone.utc)
        # 使用 UTC 时区进行本地化

        tm.assert_index_equal(dr_utc, localized)
        # 使用测试框架验证 UTC 本地化后的结果与预期结果是否相等
    # 使用 pytest 的 parametrize 装饰器，为 test_dti_tz_localize_nonexistent_shift 方法参数化多组输入
    @pytest.mark.parametrize(
        "start_ts, tz, end_ts, shift",
        [
            # 第一组参数化数据：起始时间、时区、结束时间、偏移量（向前）
            ["2015-03-29 02:20:00", "Europe/Warsaw", "2015-03-29 03:00:00", "forward"],
            # 第二组参数化数据：起始时间、时区、结束时间、偏移量（向后）
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 01:59:59.999999999",
                "backward",
            ],
            # 第三组参数化数据：起始时间、时区、结束时间、时间增量（1小时）
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 03:20:00",
                timedelta(hours=1),
            ],
            # 第四组参数化数据：起始时间、时区、结束时间、时间增量（-1小时）
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 01:20:00",
                timedelta(hours=-1),
            ],
            # 第五组参数化数据：起始时间、时区、结束时间、偏移量（向前）
            ["2018-03-11 02:33:00", "US/Pacific", "2018-03-11 03:00:00", "forward"],
            # 第六组参数化数据：起始时间、时区、结束时间、偏移量（向后）
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 01:59:59.999999999",
                "backward",
            ],
            # 第七组参数化数据：起始时间、时区、结束时间、时间增量（1小时）
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 03:33:00",
                timedelta(hours=1),
            ],
            # 第八组参数化数据：起始时间、时区、结束时间、时间增量（-1小时）
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 01:33:00",
                timedelta(hours=-1),
            ],
        ],
    )
    # 使用 pytest 的 parametrize 装饰器，为 test_dti_tz_localize_nonexistent_shift 方法参数化时区类型
    @pytest.mark.parametrize("tz_type", ["", "dateutil/"])
    # 定义测试方法 test_dti_tz_localize_nonexistent_shift，接受参数化的输入和 unit 参数
    def test_dti_tz_localize_nonexistent_shift(
        self, start_ts, tz, end_ts, shift, tz_type, unit
    ):
        # 根据 GH#8917 的说明，拼接时区类型和时区
        tz = tz_type + tz
        # 如果 shift 是字符串类型，将其格式化为 "shift_{forward/backward}"
        if isinstance(shift, str):
            shift = "shift_" + shift
        # 创建一个包含起始时间戳的 DatetimeIndex，并调整时间单位为 unit
        dti = DatetimeIndex([Timestamp(start_ts)]).as_unit(unit)
        # 执行时区本地化操作，设定不存在时的处理方式为 shift
        result = dti.tz_localize(tz, nonexistent=shift)
        # 期望结果是一个包含结束时间戳的 DatetimeIndex，并以时区 tz 本地化，并设定时间单位为 unit
        expected = DatetimeIndex([Timestamp(end_ts)]).tz_localize(tz).as_unit(unit)
        # 断言结果和期望结果相等
        tm.assert_index_equal(result, expected)

    # 使用 pytest 的 parametrize 装饰器，为 test_dti_tz_localize_nonexistent_shift_invalid 方法参数化 offset
    @pytest.mark.parametrize("offset", [-1, 1])
    # 定义测试方法 test_dti_tz_localize_nonexistent_shift_invalid，接受参数化的 offset 和 warsaw 参数
    def test_dti_tz_localize_nonexistent_shift_invalid(self, offset, warsaw):
        # 根据 GH#8917 的说明，设定时区为 warsaw
        tz = warsaw
        # 创建一个包含指定时间戳的 DatetimeIndex
        dti = DatetimeIndex([Timestamp("2015-03-29 02:20:00")])
        # 定义异常消息内容
        msg = "The provided timedelta will relocalize on a nonexistent time"
        # 使用 pytest 的断言，验证时区本地化操作在给定的时间偏移下会引发 ValueError 异常，并匹配异常消息
        with pytest.raises(ValueError, match=msg):
            dti.tz_localize(tz, nonexistent=timedelta(seconds=offset))
```