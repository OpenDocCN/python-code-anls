# `D:\src\scipysrc\pandas\pandas\tests\scalar\timestamp\methods\test_tz_localize.py`

```
    @pytest.mark.skip_ubsan
    def test_tz_localize_pushes_out_of_bounds(self):
        # 标记测试为跳过，用于 UBSan 检查
        # 测试时区本地化超出边界情况
        pytz = pytest.importorskip("pytz")  # 导入并检查是否存在 pytz 库
        msg = (
            f"Converting {Timestamp.min.strftime('%Y-%m-%d %H:%M:%S')} "
            f"underflows past {Timestamp.min}"
        )
        # 使用 US/Pacific 时区本地化最小时间戳
        pac = Timestamp.min.tz_localize(pytz.timezone("US/Pacific"))
        assert pac._value > Timestamp.min._value  # 断言本地化后时间戳大于最小时间戳
        pac.tz_convert("Asia/Tokyo")  # tz_convert 不改变数值
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            # 断言尝试使用 Asia/Tokyo 时区本地化最小时间戳抛出 OutOfBoundsDatetime 异常
            Timestamp.min.tz_localize(pytz.timezone("Asia/Tokyo"))

        msg = (
            f"Converting {Timestamp.max.strftime('%Y-%m-%d %H:%M:%S')} "
            f"overflows past {Timestamp.max}"
        )
        # 使用 Asia/Tokyo 时区本地化最大时间戳
        tokyo = Timestamp.max.tz_localize(pytz.timezone("Asia/Tokyo"))
        assert tokyo._value < Timestamp.max._value  # 断言本地化后时间戳小于最大时间戳
        tokyo.tz_convert("US/Pacific")  # tz_convert 不改变数值
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            # 断言尝试使用 US/Pacific 时区本地化最大时间戳抛出 OutOfBoundsDatetime 异常
            Timestamp.max.tz_localize(pytz.timezone("US/Pacific"))

    @pytest.mark.parametrize(
        "tz",
        [zoneinfo.ZoneInfo("US/Central"), "dateutil/US/Central", "pytz/US/Central"],
    )
    def test_tz_localize_ambiguous_bool(self, unit, tz):
        # 确保能正确接受布尔值作为模糊的参数
        # GH#14402
        if isinstance(tz, str) and tz.startswith("pytz/"):
            tz = pytz.timezone(tz.removeprefix("pytz/"))
        # 将时间戳转换为特定时间单位
        ts = Timestamp("2015-11-01 01:00:03").as_unit(unit)
        expected0 = Timestamp("2015-11-01 01:00:03-0500", tz=tz)
        expected1 = Timestamp("2015-11-01 01:00:03-0600", tz=tz)

        msg = "Cannot infer dst time from 2015-11-01 01:00:03"
        with pytest.raises(pytz.AmbiguousTimeError, match=msg):
            # 断言尝试使用模糊时间抛出 AmbiguousTimeError 异常
            ts.tz_localize(tz)

        # 使用模糊参数为 True 进行本地化
        result = ts.tz_localize(tz, ambiguous=True)
        assert result == expected0  # 断言结果符合预期值 expected0
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

        # 使用模糊参数为 False 进行本地化
        result = ts.tz_localize(tz, ambiguous=False)
        assert result == expected1  # 断言结果符合预期值 expected1
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value
    # 定义测试方法，用于测试时区本地化中的模糊性处理
    def test_tz_localize_ambiguous(self):
        # 创建一个时间戳对象，表示2014年11月2日01:00
        ts = Timestamp("2014-11-02 01:00")
        # 对时间戳进行时区本地化，处理模糊性为True，即处理模糊时间
        ts_dst = ts.tz_localize("US/Eastern", ambiguous=True)
        # 对时间戳进行时区本地化，处理模糊性为False，即不处理模糊时间
        ts_no_dst = ts.tz_localize("US/Eastern", ambiguous=False)

        # 断言：处理不模糊时间和处理模糊时间之间的差异应为3600秒（1小时）
        assert ts_no_dst._value - ts_dst._value == 3600
        
        # 准备错误消息的正则表达式，用于匹配异常消息
        msg = re.escape(
            "'ambiguous' parameter must be one of: "
            "True, False, 'NaT', 'raise' (default)"
        )
        # 使用 pytest 框架断言应该抛出 ValueError 异常，并且异常消息与预期的正则表达式匹配
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize("US/Eastern", ambiguous="infer")

        # GH#8025：测试无法本地化带时区的时间戳，应该使用 tz_convert 进行转换
        msg = "Cannot localize tz-aware Timestamp, use tz_convert for conversions"
        with pytest.raises(TypeError, match=msg):
            Timestamp("2011-01-01", tz="US/Eastern").tz_localize("Asia/Tokyo")

        # 测试无法转换非时区感知的时间戳，应该使用 tz_localize 进行本地化
        msg = "Cannot convert tz-naive Timestamp, use tz_localize to localize"
        with pytest.raises(TypeError, match=msg):
            Timestamp("2011-01-01").tz_convert("Asia/Tokyo")

    # 使用 pytest 的参数化标记，定义多个参数化测试用例
    @pytest.mark.parametrize(
        "stamp, tz",
        [
            ("2015-03-08 02:00", "US/Eastern"),
            ("2015-03-08 02:30", "US/Pacific"),
            ("2015-03-29 02:00", "Europe/Paris"),
            ("2015-03-29 02:30", "Europe/Belgrade"),
        ],
    )
    # 定义测试方法，用于测试时区本地化中不存在的时间处理
    def test_tz_localize_nonexistent(self, stamp, tz):
        # 创建一个时间戳对象
        ts = Timestamp(stamp)
        # 使用 pytest 断言应该抛出 NonExistentTimeError 异常，并且异常消息应匹配给定的时间戳
        with pytest.raises(NonExistentTimeError, match=stamp):
            ts.tz_localize(tz)
        # GH 22644：再次使用 pytest 断言应该抛出 NonExistentTimeError 异常，并且异常消息应匹配给定的时间戳
        with pytest.raises(NonExistentTimeError, match=stamp):
            ts.tz_localize(tz, nonexistent="raise")
        # 断言：使用 nonexistent="NaT" 参数本地化后应该得到 NaT（Not a Time）对象
        assert ts.tz_localize(tz, nonexistent="NaT") is NaT

    # 使用 pytest 的参数化标记，定义多个参数化测试用例
    @pytest.mark.parametrize(
        "stamp, tz, forward_expected, backward_expected",
        [
            (
                "2015-03-29 02:00:00",
                "Europe/Warsaw",
                "2015-03-29 03:00:00",
                "2015-03-29 01:59:59",
            ),  # utc+1 -> utc+2
            (
                "2023-03-12 02:00:00",
                "America/Los_Angeles",
                "2023-03-12 03:00:00",
                "2023-03-12 01:59:59",
            ),  # utc-8 -> utc-7
            (
                "2023-03-26 01:00:00",
                "Europe/London",
                "2023-03-26 02:00:00",
                "2023-03-26 00:59:59",
            ),  # utc+0 -> utc+1
            (
                "2023-03-26 00:00:00",
                "Atlantic/Azores",
                "2023-03-26 01:00:00",
                "2023-03-25 23:59:59",
            ),  # utc-1 -> utc+0
        ],
    )
    # 定义测试方法，用于测试时区本地化中不存在的时间处理和时区偏移
    def test_tz_localize_nonexistent_shift(
        self, stamp, tz, forward_expected, backward_expected
    ):
        # 创建一个时间戳对象
        ts = Timestamp(stamp)
        # 时区本地化，处理不存在时间时向前偏移
        forward_ts = ts.tz_localize(tz, nonexistent="shift_forward")
        # 断言：时区本地化后的结果应与预期的时间戳对象相等
        assert forward_ts == Timestamp(forward_expected, tz=tz)
        # 时区本地化，处理不存在时间时向后偏移
        backward_ts = ts.tz_localize(tz, nonexistent="shift_backward")
        # 断言：时区本地化后的结果应与预期的时间戳对象相等
        assert backward_ts == Timestamp(backward_expected, tz=tz)
    # 定义一个测试方法，用于测试在时区本地化过程中处理模糊时间的异常情况
    def test_tz_localize_ambiguous_raise(self):
        # 创建一个 Timestamp 对象，表示特定日期和时间
        ts = Timestamp("2015-11-1 01:00")
        # 设置匹配的异常消息，用于检查异常触发时的错误信息
        msg = "Cannot infer dst time from 2015-11-01 01:00:00,"
        # 使用 pytest 库断言会触发 AmbiguousTimeError 异常，并匹配指定的错误消息
        with pytest.raises(AmbiguousTimeError, match=msg):
            # 在特定时区（"US/Pacific"）尝试对时间戳进行本地化，指定模糊时间的处理方式为抛出异常
            ts.tz_localize("US/Pacific", ambiguous="raise")

    # 定义一个测试方法，用于测试在时区本地化过程中处理不存在时间的无效参数的情况
    def test_tz_localize_nonexistent_invalid_arg(self, warsaw):
        # 获取华沙时区的信息
        tz = warsaw
        # 创建一个 Timestamp 对象，表示特定日期和时间
        ts = Timestamp("2015-03-29 02:00:00")
        # 设置匹配的异常消息，用于检查异常触发时的错误信息
        msg = (
            "The nonexistent argument must be one of 'raise', 'NaT', "
            "'shift_forward', 'shift_backward' or a timedelta object"
        )
        # 使用 pytest 库断言会触发 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match=msg):
            # 在特定时区下（tz），尝试对时间戳进行本地化，指定不存在时间的处理方式为传入无效参数 "foo"
            ts.tz_localize(tz, nonexistent="foo")

    # 使用 pytest 的参数化标记，定义一个参数化测试方法，用于测试时区本地化的往返性
    @pytest.mark.parametrize(
        "stamp",
        [
            "2014-02-01 09:00",
            "2014-07-08 09:00",
            "2014-11-01 17:00",
            "2014-11-05 00:00",
        ],
    )
    def test_tz_localize_roundtrip(self, stamp, tz_aware_fixture):
        # 获取测试时区的信息
        tz = tz_aware_fixture
        # 创建一个 Timestamp 对象，表示特定日期和时间
        ts = Timestamp(stamp)
        # 将时间戳在指定时区下进行本地化
        localized = ts.tz_localize(tz)
        # 断言本地化后的时间戳与原始时间戳在同一时区下具有相同的值
        assert localized == Timestamp(stamp, tz=tz)

        # 设置匹配的异常消息，用于检查异常触发时的错误信息
        msg = "Cannot localize tz-aware Timestamp"
        # 使用 pytest 库断言会触发 TypeError 异常，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            # 尝试在已经有时区信息的时间戳上再次进行本地化操作
            localized.tz_localize(tz)

        # 将本地化后的时间戳重置为无时区状态，并断言重置后与原始时间戳相等，并且时区信息为 None
        reset = localized.tz_localize(None)
        assert reset == ts
        assert reset.tzinfo is None

    # 定义一个测试方法，用于验证 pytz 和 dateutil 在处理夏令时转换时的兼容性
    def test_tz_localize_ambiguous_compat(self):
        # 使用 pytest.importorskip 方法导入 pytz 库，如果导入失败则跳过测试
        pytz = pytest.importorskip("pytz")
        # 创建一个 naive Timestamp 对象，表示特定日期和时间，没有时区信息
        naive = Timestamp("2013-10-27 01:00:00")

        # 使用 pytz 创建欧洲伦敦时区的对象
        pytz_zone = pytz.timezone("Europe/London")
        # 使用 dateutil 提供的字符串表示创建欧洲伦敦时区的对象
        dateutil_zone = "dateutil/Europe/London"
        # 在不允许模糊时间的情况下，分别对 naive 时间戳进行本地化操作
        result_pytz = naive.tz_localize(pytz_zone, ambiguous=False)
        result_dateutil = naive.tz_localize(dateutil_zone, ambiguous=False)
        # 断言两种方法本地化后的时间戳数值相等，均为特定时间戳对应的时间戳值
        assert result_pytz._value == result_dateutil._value
        assert result_pytz._value == 1382835600

        # 验证在修复模糊行为后的处理方式
        assert result_pytz.to_pydatetime().tzname() == "GMT"
        assert result_dateutil.to_pydatetime().tzname() == "GMT"
        assert str(result_pytz) == str(result_dateutil)

        # 在允许模糊时间的情况下，分别对 naive 时间戳进行本地化操作
        result_pytz = naive.tz_localize(pytz_zone, ambiguous=True)
        result_dateutil = naive.tz_localize(dateutil_zone, ambiguous=True)
        # 断言两种方法本地化后的时间戳数值相等，均为特定时间戳对应的时间戳值
        assert result_pytz._value == result_dateutil._value
        assert result_pytz._value == 1382832000

        # 验证修复模糊行为后的处理方式
        assert str(result_pytz) == str(result_dateutil)
        assert (
            result_pytz.to_pydatetime().tzname()
            == result_dateutil.to_pydatetime().tzname()
        )

    # 使用 pytest 的参数化标记，定义一个参数化测试方法，用于测试不同时区信息的时区本地化
    @pytest.mark.parametrize(
        "tz",
        [
            "pytz/US/Eastern",
            gettz("US/Eastern"),
            zoneinfo.ZoneInfo("US/Eastern"),
            "dateutil/US/Eastern",
        ],
    )
    # 定义测试方法，用于测试时间戳的时区本地化功能
    def test_timestamp_tz_localize(self, tz):
        # 如果时区参数是字符串且以"pytz/"开头，则尝试导入pytz库
        if isinstance(tz, str) and tz.startswith("pytz/"):
            pytz = pytest.importorskip("pytz")
            # 从时区参数中移除"pytz/"前缀后，作为新的时区字符串
            tz = pytz.timezone(tz.removeprefix("pytz/"))
        # 创建一个时间戳对象，表示"3/11/2012 04:00"时刻
        stamp = Timestamp("3/11/2012 04:00")

        # 对时间戳进行时区本地化操作，使用给定的时区
        result = stamp.tz_localize(tz)
        # 创建一个预期的本地化时间戳对象，与上面创建的时刻相同，但使用了相同的时区
        expected = Timestamp("3/11/2012 04:00", tz=tz)
        # 断言结果的小时数与预期相同
        assert result.hour == expected.hour
        # 断言结果与预期的时间戳对象完全相同
        assert result == expected

    # 使用参数化测试装饰器标记多个参数组合，测试时区本地化的边界情况
    @pytest.mark.parametrize(
        "start_ts, tz, end_ts, shift",
        [
            ["2015-03-29 02:20:00", "Europe/Warsaw", "2015-03-29 03:00:00", "forward"],
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 01:59:59.999999999",
                "backward",
            ],
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 03:20:00",
                timedelta(hours=1),
            ],
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 01:20:00",
                timedelta(hours=-1),
            ],
            ["2018-03-11 02:33:00", "US/Pacific", "2018-03-11 03:00:00", "forward"],
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 01:59:59.999999999",
                "backward",
            ],
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 03:33:00",
                timedelta(hours=1),
            ],
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 01:33:00",
                timedelta(hours=-1),
            ],
        ],
    )
    # 再次使用参数化测试装饰器，测试时区本地化的非存在转移情况
    @pytest.mark.parametrize("tz_type", ["", "dateutil/"])
    def test_timestamp_tz_localize_nonexistent_shift(
        self, start_ts, tz, end_ts, shift, tz_type, unit
    ):
        # GitHub 上的问题编号，用于跟踪相关问题
        # 根据 tz_type 构造完整的时区字符串
        tz = tz_type + tz
        # 如果 shift 是字符串类型，则添加前缀"shift_"，以表示转移类型
        if isinstance(shift, str):
            shift = "shift_" + shift
        # 创建时间戳对象，并将其按照指定的时间单位转换
        ts = Timestamp(start_ts).as_unit(unit)
        # 对时间戳进行时区本地化，处理非存在转移情况
        result = ts.tz_localize(tz, nonexistent=shift)
        # 创建预期的本地化时间戳对象，使用相同的时区
        expected = Timestamp(end_ts).tz_localize(tz)

        # 根据不同的时间单位执行断言，确保本地化后的结果与预期相符
        if unit == "us":
            assert result == expected.replace(nanosecond=0)
        elif unit == "ms":
            micros = expected.microsecond - expected.microsecond % 1000
            assert result == expected.replace(microsecond=micros, nanosecond=0)
        elif unit == "s":
            assert result == expected.replace(microsecond=0, nanosecond=0)
        else:
            assert result == expected
        # 断言结果对象的单位属性与预期相符
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

    # 使用参数化测试装饰器，测试时区本地化的偏移情况
    @pytest.mark.parametrize("offset", [-1, 1])


这里是对给定代码的详细注释，每行代码都按照要求进行了解释和说明。
    # 定义测试方法，验证在本地化时使用不存在的时区偏移会引发无效偏移错误
    def test_timestamp_tz_localize_nonexistent_shift_invalid(self, offset, warsaw):
        # 设定时区为华沙时区
        tz = warsaw
        # 创建时间戳对象，表示 "2015-03-29 02:20:00"
        ts = Timestamp("2015-03-29 02:20:00")
        # 错误消息，指示 timedelta 参数会在不存在的时间上重新本地化
        msg = "The provided timedelta will relocalize on a nonexistent time"
        # 使用 pytest 断言，验证在重新本地化时会引发 ValueError 异常，并匹配特定错误消息
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize(tz, nonexistent=timedelta(seconds=offset))

    # 定义测试方法，验证在本地化时使用不存在的时区偏移会返回 NaT（Not a Time）
    def test_timestamp_tz_localize_nonexistent_NaT(self, warsaw, unit):
        # 设定时区为华沙时区
        tz = warsaw
        # 创建时间戳对象，表示 "2015-03-29 02:20:00"，并将其转换为指定单位
        ts = Timestamp("2015-03-29 02:20:00").as_unit(unit)
        # 调用 tz_localize 方法，使用 nonexistent="NaT" 参数进行本地化
        result = ts.tz_localize(tz, nonexistent="NaT")
        # 使用断言验证结果是否为 NaT
        assert result is NaT

    # 定义测试方法，验证在本地化时使用不存在的时区偏移会引发非存在时间错误
    def test_timestamp_tz_localize_nonexistent_raise(self, warsaw, unit):
        # 设定时区为华沙时区
        tz = warsaw
        # 创建时间戳对象，表示 "2015-03-29 02:20:00"，并将其转换为指定单位
        ts = Timestamp("2015-03-29 02:20:00").as_unit(unit)
        # 预期错误消息，指示非存在时间错误
        msg = "2015-03-29 02:20:00"
        # 使用 pytest 断言，验证在重新本地化时会引发 pytz.NonExistentTimeError 异常，并匹配特定错误消息
        with pytest.raises(pytz.NonExistentTimeError, match=msg):
            ts.tz_localize(tz, nonexistent="raise")
        # 错误消息，指示 nonexistent 参数必须是 'raise', 'NaT', 'shift_forward', 'shift_backward' 或 timedelta 对象之一
        msg = (
            "The nonexistent argument must be one of 'raise', 'NaT', "
            "'shift_forward', 'shift_backward' or a timedelta object"
        )
        # 使用 pytest 断言，验证在重新本地化时会引发 ValueError 异常，并匹配特定错误消息
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize(tz, nonexistent="foo")
```