# `D:\src\scipysrc\pandas\pandas\tests\scalar\timestamp\methods\test_replace.py`

```
    # 导入需要的模块和类
    from datetime import datetime
    import zoneinfo

    from dateutil.tz import gettz
    import numpy as np
    import pytest

    # 导入 pandas 相关模块和类
    from pandas._libs.tslibs import (
        OutOfBoundsDatetime,
        Timestamp,
        conversion,
    )
    from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
    from pandas.compat import WASM
    import pandas.util._test_decorators as td

    import pandas._testing as tm


    class TestTimestampReplace:
        def test_replace_out_of_pydatetime_bounds(self):
            # 创建 Timestamp 对象，设置时间单位为纳秒
            ts = Timestamp("2016-01-01").as_unit("ns")

            # 准备错误消息字符串
            msg = "Out of bounds timestamp: 99999-01-01 00:00:00 with frequency 'ns'"
            # 使用 pytest 断言捕获 OutOfBoundsDatetime 异常并验证错误消息
            with pytest.raises(OutOfBoundsDatetime, match=msg):
                ts.replace(year=99_999)

            # 将时间单位转换为毫秒
            ts = ts.as_unit("ms")
            # 替换 Timestamp 对象的年份为 99999
            result = ts.replace(year=99_999)
            # 验证替换后的年份和内部值是否符合预期
            assert result.year == 99_999
            assert result._value == Timestamp(np.datetime64("99999-01-01", "ms"))._value

        def test_replace_non_nano(self):
            # 创建 Timestamp 对象，设置时间单位为微秒
            ts = Timestamp._from_value_and_reso(
                91514880000000000, NpyDatetimeUnit.NPY_FR_us.value, None
            )
            # 验证 Timestamp 对象转换为 Python datetime 对象是否符合预期日期
            assert ts.to_pydatetime() == datetime(4869, 12, 28)

            # 替换 Timestamp 对象的年份为 4900
            result = ts.replace(year=4900)
            # 验证替换后的年份和分辨率是否保持不变，并且日期是否符合预期
            assert result._creso == ts._creso
            assert result.to_pydatetime() == datetime(4900, 12, 28)

        def test_replace_naive(self):
            # 创建 naive 的 Timestamp 对象
            ts = Timestamp("2016-01-01 09:00:00")
            # 替换 Timestamp 对象的小时为 0
            result = ts.replace(hour=0)
            # 预期结果为日期部分不变，时间变为 00:00:00
            expected = Timestamp("2016-01-01 00:00:00")
            assert result == expected

        def test_replace_aware(self, tz_aware_fixture):
            tz = tz_aware_fixture
            # 创建带时区信息的 Timestamp 对象
            ts = Timestamp("2016-01-01 09:00:00", tz=tz)
            # 替换 Timestamp 对象的小时为 0
            result = ts.replace(hour=0)
            # 预期结果为日期和时区信息不变，时间变为 00:00:00
            expected = Timestamp("2016-01-01 00:00:00", tz=tz)
            assert result == expected

        def test_replace_preserves_nanos(self, tz_aware_fixture):
            tz = tz_aware_fixture
            # 创建带时区信息和纳秒的 Timestamp 对象
            ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
            # 替换 Timestamp 对象的小时为 0
            result = ts.replace(hour=0)
            # 预期结果为日期、纳秒和时区信息不变，时间变为 00:00:00.000000123
            expected = Timestamp("2016-01-01 00:00:00.000000123", tz=tz)
            assert result == expected

        def test_replace_multiple(self, tz_aware_fixture):
            tz = tz_aware_fixture
            # 创建带时区信息和纳秒的 Timestamp 对象
            ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
            # 替换 Timestamp 对象的多个时间组成部分
            result = ts.replace(
                year=2015,
                month=2,
                day=2,
                hour=0,
                minute=5,
                second=5,
                microsecond=5,
                nanosecond=5,
            )
            # 预期结果为替换后的完整日期和时间，保持原有的纳秒和时区信息
            expected = Timestamp("2015-02-02 00:05:05.000005005", tz=tz)
            assert result == expected
    # 定义一个测试方法，用于测试在给定的时区下替换无效关键字参数的行为
    def test_replace_invalid_kwarg(self, tz_aware_fixture):
        # 获取时区装置的实例
        tz = tz_aware_fixture
        # 创建一个带有时区信息的时间戳对象，指定特定日期和时间
        ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
        # 定义一个错误消息，用于匹配在调用 replace() 方法时出现的异常情况
        msg = r"replace\(\) got an unexpected keyword argument"
        # 使用 pytest 断言来验证是否引发了预期的 TypeError 异常，并检查错误消息是否匹配
        with pytest.raises(TypeError, match=msg):
            ts.replace(foo=5)

    # 定义一个测试方法，用于测试在给定的时区下替换整数参数的行为
    def test_replace_integer_args(self, tz_aware_fixture):
        # 获取时区装置的实例
        tz = tz_aware_fixture
        # 创建一个带有时区信息的时间戳对象，指定特定日期和时间
        ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
        # 定义一个错误消息，用于匹配在调用 replace() 方法时出现的异常情况
        msg = "value must be an integer, received <class 'float'> for hour"
        # 使用 pytest 断言来验证是否引发了预期的 ValueError 异常，并检查错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            ts.replace(hour=0.1)

    # 使用 pytest.mark.skipif 装饰器标记的测试方法，用于测试在特定条件下替换时区信息等效于将时区信息替换为 None 的行为
    @pytest.mark.skipif(WASM, reason="tzset is not available on WASM")
    def test_replace_tzinfo_equiv_tz_localize_none(self):
        # 创建一个带有时区信息的时间戳对象，指定特定日期、时间和时区
        ts = Timestamp("2013-11-03 01:59:59.999999-0400", tz="US/Eastern")
        # 断言将时间戳本地化为无时区信息的结果等同于将时区信息替换为 None 的结果
        assert ts.tz_localize(None) == ts.replace(tzinfo=None)

    # 使用 td.skip_if_windows 和 pytest.mark.skipif 装饰器标记的测试方法，用于测试在给定时区下替换时区信息的行为
    @td.skip_if_windows
    @pytest.mark.skipif(WASM, reason="tzset is not available on WASM")
    def test_replace_tzinfo(self):
        # 创建一个标准的 Python datetime 对象，指定特定日期和时间
        dt = datetime(2016, 3, 27, 1, fold=1)
        # 获取该 datetime 对象在欧洲柏林时区的时区信息
        tzinfo = dt.astimezone(zoneinfo.ZoneInfo("Europe/Berlin")).tzinfo

        # 使用指定的时区信息替换 datetime 对象的时区信息，并与使用 pandas Timestamp 替换的结果进行比较
        result_dt = dt.replace(tzinfo=tzinfo)
        result_pd = Timestamp(dt).replace(tzinfo=tzinfo)

        # 使用 pytest 设置时区为 UTC，并验证转换后的 timestamp 是否一致
        with tm.set_timezone("UTC"):
            assert result_dt.timestamp() == result_pd.timestamp()

        # 断言 datetime 对象与 pandas Timestamp 对象在替换时区信息后的日期时间表示是否一致
        assert result_dt == result_pd
        assert result_dt == result_pd.to_pydatetime()

        # 在将时区信息替换为 None 后，再次验证 datetime 对象和 pandas Timestamp 对象的行为是否一致
        result_dt = dt.replace(tzinfo=tzinfo).replace(tzinfo=None)
        result_pd = Timestamp(dt).replace(tzinfo=tzinfo).replace(tzinfo=None)

        # 使用 pytest 设置时区为 UTC，并验证转换后的 timestamp 是否一致
        with tm.set_timezone("UTC"):
            assert result_dt.timestamp() == result_pd.timestamp()

        # 再次断言 datetime 对象与 pandas Timestamp 对象在替换时区信息后的日期时间表示是否一致
        assert result_dt == result_pd
        assert result_dt == result_pd.to_pydatetime()

    # 使用 pytest.mark.parametrize 装饰器标记的测试方法，用于测试不同时区和规范化操作的行为
    @pytest.mark.parametrize(
        "tz, normalize",
        [
            ("pytz/US/Eastern", lambda x: x.tzinfo.normalize(x)),
            (gettz("US/Eastern"), lambda x: x),
        ],
    )
    def test_replace_across_dst(self, tz, normalize):
        # GH#18319 检查：1）时区是否正确标准化；2）标准化过程中小时是否正确保留
        if isinstance(tz, str) and tz.startswith("pytz/"):
            # 如果时区是以 "pytz/" 开头的字符串，则导入 pytest 和 pytz 模块
            pytz = pytest.importorskip("pytz")
            # 从字符串中移除 "pytz/" 前缀，获取时区名称
            tz = pytz.timezone(tz.removeprefix("pytz/"))
        # 创建一个本地化的 naive 时间戳
        ts_naive = Timestamp("2017-12-03 16:03:30")
        # 使用给定的时区将 naive 时间戳本地化为 aware 时间戳
        ts_aware = conversion.localize_pydatetime(ts_naive, tz)

        # 初步的健全性检查，确认本地化后的时间戳是否保持不变
        assert ts_aware == normalize(ts_aware)

        # 在夏令时边界上执行替换
        ts2 = ts_aware.replace(month=6)

        # 检查 `replace` 方法是否保留了小时字面值
        assert (ts2.hour, ts2.minute) == (ts_aware.hour, ts_aware.minute)

        # 检查替换后的对象是否适当地被标准化
        ts2b = normalize(ts2)
        assert ts2 == ts2b

    def test_replace_dst_border(self, unit):
        # Gh 7825
        # 创建一个带时区信息的时间戳对象
        t = Timestamp("2013-11-3", tz="America/Chicago").as_unit(unit)
        # 替换时间戳的小时为 3
        result = t.replace(hour=3)
        # 期望的结果时间戳对象
        expected = Timestamp("2013-11-3 03:00:00", tz="America/Chicago")
        # 断言结果与期望是否相等
        assert result == expected
        # 检查结果对象的 `_creso` 属性是否与指定单位相关联
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

    @pytest.mark.parametrize("fold", [0, 1])
    @pytest.mark.parametrize("tz", ["dateutil/Europe/London", "Europe/London"])
    def test_replace_dst_fold(self, fold, tz, unit):
        # GH 25017
        # 创建一个 datetime 对象
        d = datetime(2019, 10, 27, 2, 30)
        # 使用指定时区创建一个带时区信息的时间戳对象，并按照指定单位转换
        ts = Timestamp(d, tz=tz).as_unit(unit)
        # 替换时间戳的小时为 1，并设置 fold 属性
        result = ts.replace(hour=1, fold=fold)
        # 创建一个期望的时间戳对象，进行本地化
        expected = Timestamp(datetime(2019, 10, 27, 1, 30)).tz_localize(
            tz, ambiguous=not fold
        )
        # 断言结果与期望是否相等
        assert result == expected
        # 检查结果对象的 `_creso` 属性是否与指定单位相关联
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

    @pytest.mark.parametrize("fold", [0, 1])
    def test_replace_preserves_fold(self, fold):
        # GH#37610 检查：replace 方法是否保留了 Timestamp 的 fold 属性
        # 获取时区对象
        tz = gettz("Europe/Moscow")

        # 创建一个带有 fold 属性的 Timestamp 对象
        ts = Timestamp(
            year=2009, month=10, day=25, hour=2, minute=30, fold=fold, tzinfo=tz
        )
        # 替换时间戳的秒为 1
        ts_replaced = ts.replace(second=1)

        # 断言替换后的时间戳的 fold 属性是否与原始时间戳相同
        assert ts_replaced.fold == fold
```