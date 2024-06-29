# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_timezones.py`

```
# 引入所需的模块和类
"""
Tests for DatetimeIndex timezone-related methods
"""
from datetime import (
    datetime,            # 日期时间类
    timedelta,           # 时间间隔类
    timezone,            # 时区类
    tzinfo,              # 时区信息基类
)
import zoneinfo          # 加载时区信息

from dateutil.tz import gettz  # 导入获取时区的函数
import numpy as np             # 引入 NumPy 库
import pytest                  # 导入 pytest 测试框架

from pandas._libs.tslibs import (
    conversion,       # 时间序列转换函数
    timezones,        # 时间序列时区管理
)

import pandas as pd         # 引入 Pandas 库并简写为 pd
from pandas import (
    DatetimeIndex,          # Pandas 时间索引类
    Timestamp,              # Pandas 时间戳类
    bdate_range,            # 工作日范围生成函数
    date_range,             # 日期范围生成函数
    isna,                   # 判断是否为缺失值函数
    to_datetime,            # 转换为时间日期函数
)
import pandas._testing as tm  # Pandas 测试工具模块

# 自定义固定偏移时区类
class FixedOffset(tzinfo):
    """Fixed offset in minutes east from UTC."""

    def __init__(self, offset, name) -> None:
        self.__offset = timedelta(minutes=offset)  # 初始化偏移量
        self.__name = name                        # 初始化时区名称

    def utcoffset(self, dt):                      # 返回UTC偏移量
        return self.__offset

    def tzname(self, dt):                        # 返回时区名称
        return self.__name

    def dst(self, dt):                           # 返回夏令时偏移量
        return timedelta(0)

fixed_off_no_name = FixedOffset(-330, None)      # 创建一个无名称的偏移量对象


class TestDatetimeIndexTimezones:
    # -------------------------------------------------------------
    # Unsorted

    def test_dti_drop_dont_lose_tz(self):
        # GH#2621 测试索引删除操作不会丢失时区信息
        ind = date_range("2012-12-01", periods=10, tz="utc")  # 创建UTC时区的日期范围索引
        ind = ind.drop(ind[-1])                                # 删除最后一个元素

        assert ind.tz is not None  # 断言索引仍然具有时区信息

    def test_dti_tz_conversion_freq(self, tz_naive_fixture):
        # GH25241 测试时间索引的时区转换和频率保持不变
        t3 = DatetimeIndex(["2019-01-01 10:00"], freq="h")                  # 创建具有小时频率的时间索引
        assert t3.tz_localize(tz=tz_naive_fixture).freq == t3.freq           # 断言本地化后的频率与原始频率相同
        t4 = DatetimeIndex(["2019-01-02 12:00"], tz="UTC", freq="min")       # 创建具有分钟频率和UTC时区的时间索引
        assert t4.tz_convert(tz="UTC").freq == t4.freq                       # 断言转换后的频率与原始频率相同

    def test_drop_dst_boundary(self):
        # see gh-18031 测试在夏令时变更时删除索引元素的行为
        tz = "Europe/Brussels"                          # 定义欧洲布鲁塞尔时区
        freq = "15min"                                  # 定义15分钟的频率

        start = Timestamp("201710290100", tz=tz)        # 定义开始时间戳
        end = Timestamp("201710290300", tz=tz)          # 定义结束时间戳
        index = date_range(start=start, end=end, freq=freq)  # 创建日期范围索引

        expected = DatetimeIndex(                       # 创建预期的日期时间索引对象
            [
                "201710290115",
                "201710290130",
                "201710290145",
                "201710290200",
                "201710290215",
                "201710290230",
                "201710290245",
                "201710290200",
                "201710290215",
                "201710290230",
                "201710290245",
                "201710290300",
            ],
            dtype="M8[ns, Europe/Brussels]",            # 指定数据类型为带时区的日期时间
            freq=freq,                                  # 设置频率
            ambiguous=[                                 # 设置模棱两可的时间戳列表
                True, True, True, True, True, True, True,
                False, False, False, False, False,
            ],
        )
        result = index.drop(index[0])                   # 删除索引中的第一个元素
        tm.assert_index_equal(result, expected)          # 断言结果与预期相等
    # 测试本地化日期范围函数，接受一个时间单位参数
    def test_date_range_localize(self, unit):
        # 创建一个日期范围，从 "3/11/2012 03:00" 开始，包含 15 个小时，频率为每小时一次，时区为 "US/Eastern"，时间单位为 unit
        rng = date_range(
            "3/11/2012 03:00", periods=15, freq="h", tz="US/Eastern", unit=unit
        )
        # 创建一个带有时区信息的日期时间索引，指定了具体的数据类型
        rng2 = DatetimeIndex(
            ["3/11/2012 03:00", "3/11/2012 04:00"], dtype=f"M8[{unit}, US/Eastern]"
        )
        # 创建另一个日期范围，与第一个日期范围类似，但没有指定时区
        rng3 = date_range("3/11/2012 03:00", periods=15, freq="h", unit=unit)
        # 将第三个日期范围本地化到 "US/Eastern" 时区
        rng3 = rng3.tz_localize("US/Eastern")

        # 使用测试工具函数检查两个日期范围对象是否相等
        tm.assert_index_equal(rng._with_freq(None), rng3)

        # 检查第一个日期范围对象的第一个时间戳，确认其小时部分为 3
        val = rng[0]
        # 创建一个预期的时间戳对象，时区为 "US/Eastern"，小时部分为 3
        exp = Timestamp("3/11/2012 03:00", tz="US/Eastern")

        # 断言验证第一个时间戳的小时部分是否为 3
        assert val.hour == 3
        # 断言验证预期时间戳的小时部分是否为 3
        assert exp.hour == 3
        # 断言验证两个时间戳对象是否具有相同的 UTC 值
        assert val == exp
        # 使用测试工具函数检查两个日期范围对象的前两个元素是否相等
        tm.assert_index_equal(rng[:2], rng2)

    # 测试本地化日期范围函数的另一个案例
    def test_date_range_localize2(self, unit):
        # 创建一个日期范围，从 "3/11/2012 00:00" 开始，包含 2 个小时，频率为每小时一次，时区为 "US/Eastern"，时间单位为 unit
        rng = date_range(
            "3/11/2012 00:00", periods=2, freq="h", tz="US/Eastern", unit=unit
        )
        # 创建一个带有时区信息的日期时间索引，指定了具体的数据类型和频率
        rng2 = DatetimeIndex(
            ["3/11/2012 00:00", "3/11/2012 01:00"],
            dtype=f"M8[{unit}, US/Eastern]",
            freq="h",
        )
        # 使用测试工具函数检查两个日期范围对象是否相等
        tm.assert_index_equal(rng, rng2)
        # 创建一个预期的时间戳对象，时区为 "US/Eastern"，小时部分为 0
        exp = Timestamp("3/11/2012 00:00", tz="US/Eastern")
        # 断言验证预期时间戳的小时部分是否为 0
        assert exp.hour == 0
        # 断言验证第一个日期范围对象的第一个时间戳是否与预期时间戳相等
        assert rng[0] == exp
        # 创建另一个预期的时间戳对象，时区为 "US/Eastern"，小时部分为 1
        exp = Timestamp("3/11/2012 01:00", tz="US/Eastern")
        # 断言验证预期时间戳的小时部分是否为 1
        assert exp.hour == 1
        # 断言验证第一个日期范围对象的第二个时间戳是否与预期时间戳相等
        assert rng[1] == exp

        # 创建一个日期范围，从 "3/11/2012 00:00" 开始，包含 10 个小时，频率为每小时一次，时区为 "US/Eastern"，时间单位为 unit
        rng = date_range(
            "3/11/2012 00:00", periods=10, freq="h", tz="US/Eastern", unit=unit
        )
        # 断言验证第三个日期范围对象的第三个时间戳的小时部分是否为 3
        assert rng[2].hour == 3

    # 测试不同时区下时间戳的相等性
    def test_timestamp_equality_different_timezones(self):
        # 创建一个 UTC 时区的日期范围
        utc_range = date_range("1/1/2000", periods=20, tz="UTC")
        # 将日期范围转换到 "US/Eastern" 时区
        eastern_range = utc_range.tz_convert("US/Eastern")
        # 将日期范围转换到 "Europe/Berlin" 时区
        berlin_range = utc_range.tz_convert("Europe/Berlin")

        # 使用 zip 函数逐一比较三个日期范围对象对应位置的时间戳是否相等
        for a, b, c in zip(utc_range, eastern_range, berlin_range):
            assert a == b
            assert b == c
            assert a == c

        # 断言验证整个日期范围对象在不同时区转换后的时间戳是否全部相等
        assert (utc_range == eastern_range).all()
        assert (utc_range == berlin_range).all()
        assert (berlin_range == eastern_range).all()

    # 测试带时区的日期时间索引对象的相等性
    def test_dti_equals_with_tz(self):
        # 创建一个带有 UTC 时区的日期范围对象
        left = date_range("1/1/2011", periods=100, freq="h", tz="utc")
        # 创建一个带有 "US/Eastern" 时区的日期范围对象
        right = date_range("1/1/2011", periods=100, freq="h", tz="US/Eastern")

        # 断言验证两个日期范围对象是否不相等
        assert not left.equals(right)

    # 使用参数化测试框架，测试带有不同时区字符串的日期时间索引对象
    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_tz_nat(self, tzstr):
        # 创建一个日期时间索引对象，包含一个具有指定时区的时间戳和一个 NaT（Not a Time）对象
        idx = DatetimeIndex([Timestamp("2013-1-1", tz=tzstr), pd.NaT])

        # 断言验证索引对象的第二个元素是否为 NaT
        assert isna(idx[1])
        # 断言验证索引对象的第一个时间戳是否包含有效的时区信息
        assert idx[0].tzinfo is not None

    # 使用参数化测试框架，测试带有不同时区字符串的日期时间索引对象
    @pytest.mark.parametrize("tzstr", ["pytz/US/Eastern", "dateutil/US/Eastern"])
    # 定义测试方法，用于验证处理 UTC 时间戳和本地化的功能
    def test_utc_box_timestamp_and_localize(self, tzstr):
        # 检查时区字符串是否以 "pytz/" 开头，如果是则尝试导入 "pytz" 库并移除前缀
        if tzstr.startswith("pytz/"):
            pytest.importorskip("pytz")
            tzstr = tzstr.removeprefix("pytz/")
        
        # 根据时区字符串获取时区对象
        tz = timezones.maybe_get_tz(tzstr)

        # 创建一个 UTC 时区下的日期范围，频率为每小时
        rng = date_range("3/11/2012", "3/12/2012", freq="h", tz="utc")
        # 将日期范围转换为指定时区的日期范围
        rng_eastern = rng.tz_convert(tzstr)

        # 获取预期的本地化时间，即 UTC 时区下日期范围的最后一个时间点的本地化
        expected = rng[-1].astimezone(tz)

        # 获取日期范围在指定时区下的最后一个时间点
        stamp = rng_eastern[-1]
        # 断言最后一个时间点与预期相等
        assert stamp == expected
        # 断言最后一个时间点的时区信息与预期的时区信息相同
        assert stamp.tzinfo == expected.tzinfo

        # 创建另一个 UTC 时区下的日期范围，频率为每小时
        rng = date_range("3/13/2012", "3/14/2012", freq="h", tz="utc")
        # 将日期范围转换为指定时区的日期范围
        rng_eastern = rng.tz_convert(tzstr)
        # 断言日期范围的第一个时间点的时区信息包含 'EDT' 或 'tzfile'
        assert "EDT" in repr(rng_eastern[0].tzinfo) or "tzfile" in repr(
            rng_eastern[0].tzinfo
        )

    # 使用参数化标记，测试带有指定时区的功能
    @pytest.mark.parametrize(
        "tz", [zoneinfo.ZoneInfo("US/Central"), gettz("US/Central")]
    )
    def test_with_tz(self, tz):
        # 确保处理日期范围的时区为 UTC
        start = datetime(2011, 3, 12, tzinfo=timezone.utc)
        dr = bdate_range(start, periods=50, freq=pd.offsets.Hour())
        assert dr.tz is timezone.utc

        # 创建一个包含本地时间的日期范围
        dr = bdate_range("1/1/2005", "1/1/2009", tz=timezone.utc)
        # 使用指定的时区创建日期范围
        dr = bdate_range("1/1/2005", "1/1/2009", tz=tz)

        # 将日期范围转换为指定的中央时区
        central = dr.tz_convert(tz)
        assert central.tz is tz

        # 获取日期范围中第一个时间点的本地时间，去除时区信息后再转换为本地化的日期时间对象
        naive = central[0].to_pydatetime().replace(tzinfo=None)
        comp = conversion.localize_pydatetime(naive, tz).tzinfo
        assert central[0].tz is comp

        # 比较与本地化时区的日期时间对象
        naive = dr[0].to_pydatetime().replace(tzinfo=None)
        comp = conversion.localize_pydatetime(naive, tz).tzinfo
        assert central[0].tz is comp

        # 创建具有时区信息的日期范围
        dr = bdate_range(
            datetime(2005, 1, 1, tzinfo=timezone.utc),
            datetime(2009, 1, 1, tzinfo=timezone.utc),
        )
        # 断言当开始日期和结束日期同时具有不同时区时会引发异常
        msg = "Start and end cannot both be tz-aware with different timezones"
        with pytest.raises(Exception, match=msg):
            bdate_range(datetime(2005, 1, 1, tzinfo=timezone.utc), "1/1/2009", tz=tz)

    # 使用参数化标记，测试将时区感知的日期时间对象转换为不同时区的功能
    @pytest.mark.parametrize(
        "tz", [zoneinfo.ZoneInfo("US/Eastern"), gettz("US/Eastern")]
    )
    def test_dti_convert_tz_aware_datetime_datetime(self, tz):
        # GH#1581
        # 创建一组日期时间对象
        dates = [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)]

        # 将日期时间对象本地化为指定时区，并转换为纳秒单位的日期时间索引
        dates_aware = [conversion.localize_pydatetime(x, tz) for x in dates]
        result = DatetimeIndex(dates_aware).as_unit("ns")
        # 断言结果的时区与预期的时区相同
        assert timezones.tz_compare(result.tz, tz)

        # 将本地化的日期时间对象转换为 UTC 时区，并转换为纳秒单位的日期时间索引
        converted = to_datetime(dates_aware, utc=True).as_unit("ns")
        ex_vals = np.array([Timestamp(x).as_unit("ns")._value for x in dates_aware])
        # 断言转换后的时间戳数组与预期值相等
        tm.assert_numpy_array_equal(converted.asi8, ex_vals)
        # 断言转换后的日期时间索引的时区为 UTC
        assert converted.tz is timezone.utc
```