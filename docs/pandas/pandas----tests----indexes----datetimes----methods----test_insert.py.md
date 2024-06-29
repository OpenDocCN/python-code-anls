# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_insert.py`

```
    # 导入必要的模块和类
    from datetime import datetime
    import zoneinfo

    import numpy as np
    import pytest

    from pandas import (
        NA,
        DatetimeIndex,
        Index,
        NaT,
        Timestamp,
        date_range,
    )
    import pandas._testing as tm


    class TestInsert:
        # 使用 pytest 的参数化装饰器，测试各种空值情况
        @pytest.mark.parametrize("null", [None, np.nan, np.datetime64("NaT"), NaT, NA])
        @pytest.mark.parametrize("tz", [None, "UTC", "US/Eastern"])
        def test_insert_nat(self, tz, null):
            # 测试用例参考 GH#16537, GH#18295 (未列出)

            # 创建带有时区信息的日期时间索引
            idx = DatetimeIndex(["2017-01-01"], tz=tz)
            # 预期结果是在索引最前面插入空值后的索引
            expected = DatetimeIndex(["NaT", "2017-01-01"], tz=tz)
            # 如果指定了时区且空值是 numpy.datetime64 类型，则预期结果是一个对象类型的索引
            if tz is not None and isinstance(null, np.datetime64):
                expected = Index([null, idx[0]], dtype=object)

            # 执行插入操作
            res = idx.insert(0, null)
            # 断言插入后的结果与预期结果相同
            tm.assert_index_equal(res, expected)

        @pytest.mark.parametrize("tz", [None, "UTC", "US/Eastern"])
        def test_insert_invalid_na(self, tz):
            # 创建带有时区信息的日期时间索引
            idx = DatetimeIndex(["2017-01-01"], tz=tz)

            # 创建一个 NaT 的 timedelta64 对象
            item = np.timedelta64("NaT")
            # 执行插入操作
            result = idx.insert(0, item)
            # 预期结果是在索引最前面插入 NaT 后的索引
            expected = Index([item] + list(idx), dtype=object)
            # 断言插入后的结果与预期结果相同
            tm.assert_index_equal(result, expected)

        def test_insert_empty_preserves_freq(self, tz_naive_fixture):
            # 测试用例参考 GH#33573
            tz = tz_naive_fixture
            # 创建一个空的日期时间索引，带有特定的频率
            dti = DatetimeIndex([], tz=tz, freq="D")
            # 创建一个带有时区信息的 Timestamp 对象
            item = Timestamp("2017-04-05").tz_localize(tz)

            # 执行插入操作
            result = dti.insert(0, item)
            # 断言插入后的结果的频率与原始索引的频率相同
            assert result.freq == dti.freq

            # 当插入不符合频率要求的对象时，预期结果的频率为 None
            dti = DatetimeIndex([], tz=tz, freq="W-THU")
            result = dti.insert(0, item)
            assert result.freq is None

        def test_insert(self, unit):
            # 创建一个带有单位的日期时间索引
            idx = DatetimeIndex(
                ["2000-01-04", "2000-01-01", "2000-01-02"], name="idx"
            ).as_unit(unit)

            # 执行插入操作
            result = idx.insert(2, datetime(2000, 1, 5))
            # 创建预期的日期时间索引
            exp = DatetimeIndex(
                ["2000-01-04", "2000-01-01", "2000-01-05", "2000-01-02"], name="idx"
            ).as_unit(unit)
            # 断言插入后的结果与预期结果相同
            tm.assert_index_equal(result, exp)

            # 当插入非日期时间对象时，预期结果是一个对象类型的索引
            result = idx.insert(1, "inserted")
            expected = Index(
                [
                    datetime(2000, 1, 4),
                    "inserted",
                    datetime(2000, 1, 1),
                    datetime(2000, 1, 2),
                ],
                name="idx",
            )
            assert not isinstance(result, DatetimeIndex)
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
    def test_insert2(self, unit):
        # 使用 date_range 函数创建一个时间索引 idx，从 "1/1/2000" 开始，周期为 3，频率为 "ME"，名称为 "idx"，单位为 unit
        idx = date_range("1/1/2000", periods=3, freq="ME", name="idx", unit=unit)

        # 保留频率
        # 创建预期结果 expected_0，作为 DatetimeIndex 对象，包含指定日期列表和频率 "ME"
        expected_0 = DatetimeIndex(
            ["1999-12-31", "2000-01-31", "2000-02-29", "2000-03-31"],
            name="idx",
            freq="ME",
        ).as_unit(unit)
        
        # 创建预期结果 expected_3，作为 DatetimeIndex 对象，包含指定日期列表和频率 "ME"
        expected_3 = DatetimeIndex(
            ["2000-01-31", "2000-02-29", "2000-03-31", "2000-04-30"],
            name="idx",
            freq="ME",
        ).as_unit(unit)

        # 重置频率为 None
        # 创建预期结果 expected_1_nofreq，作为 DatetimeIndex 对象，包含指定日期列表和频率 None
        expected_1_nofreq = DatetimeIndex(
            ["2000-01-31", "2000-01-31", "2000-02-29", "2000-03-31"],
            name="idx",
            freq=None,
        ).as_unit(unit)
        
        # 创建预期结果 expected_3_nofreq，作为 DatetimeIndex 对象，包含指定日期列表和频率 None
        expected_3_nofreq = DatetimeIndex(
            ["2000-01-31", "2000-02-29", "2000-03-31", "2000-01-02"],
            name="idx",
            freq=None,
        ).as_unit(unit)

        cases = [
            # 测试用例列表，包含不同的插入位置 n、日期 d、预期结果 expected
            (0, datetime(1999, 12, 31), expected_0),
            (-3, datetime(1999, 12, 31), expected_0),
            (3, datetime(2000, 4, 30), expected_3),
            (1, datetime(2000, 1, 31), expected_1_nofreq),
            (3, datetime(2000, 1, 2), expected_3_nofreq),
        ]

        for n, d, expected in cases:
            # 对 idx 执行插入操作，插入位置为 n，插入日期为 d
            result = idx.insert(n, d)
            # 断言插入后的结果与预期结果 expected 相等
            tm.assert_index_equal(result, expected)
            # 断言插入后的结果名称与预期结果相同
            assert result.name == expected.name
            # 断言插入后的结果频率与预期结果相同
            assert result.freq == expected.freq

    def test_insert3(self, unit):
        # 使用 date_range 函数创建一个时间索引 idx，从 "1/1/2000" 开始，周期为 3，频率为 "ME"，名称为 "idx"，单位为 unit
        idx = date_range("1/1/2000", periods=3, freq="ME", name="idx", unit=unit)

        # 重置频率为 None
        # 对 idx 执行插入操作，在位置 3 插入日期为 datetime(2000, 1, 2)
        result = idx.insert(3, datetime(2000, 1, 2))
        # 创建预期结果 expected，作为 DatetimeIndex 对象，包含指定日期列表和频率 None
        expected = DatetimeIndex(
            ["2000-01-31", "2000-02-29", "2000-03-31", "2000-01-02"],
            name="idx",
            freq=None,
        ).as_unit(unit)
        # 断言插入后的结果与预期结果 expected 相等
        tm.assert_index_equal(result, expected)
        # 断言插入后的结果名称与预期结果相同
        assert result.name == expected.name
        # 断言插入后的结果频率为 None
        assert result.freq is None

    @pytest.mark.parametrize("tz", ["US/Pacific", "Asia/Singapore"])
    @pytest.mark.parametrize(
        "to_ts",
        [lambda x: x, lambda x: x.to_pydatetime()],
        ids=["Timestamp", "datetime"],
    )
    def test_insert4(self, unit, tz, to_ts):
        # 使用 date_range 函数创建一个时间索引 idx，从 "1/1/2000 09:00" 开始，周期为 6，频率为 "h"，时区为 tz，名称为 "idx"，单位为 unit
        idx = date_range(
            "1/1/2000 09:00", periods=6, freq="h", tz=tz, name="idx", unit=unit
        )
        
        # 保留频率
        # 创建预期结果 expected，作为 DatetimeIndex 对象，包含指定日期列表、频率 "h"、时区 tz，名称为 "idx"，单位为 unit
        expected = date_range(
            "1/1/2000 09:00", periods=7, freq="h", tz=tz, name="idx", unit=unit
        )
        # 创建时区对象 tz
        tz = zoneinfo.ZoneInfo(tz)
        # 将 "2000-01-01 15:00" 转换为 Timestamp 对象，并应用时区 tz
        d = to_ts(Timestamp("2000-01-01 15:00", tz=tz))
        # 对 idx 执行插入操作，在位置 6 插入日期 d
        result = idx.insert(6, d)
        # 断言插入后的结果与预期结果 expected 相等
        tm.assert_index_equal(result, expected)
        # 断言插入后的结果名称与预期结果相同
        assert result.name == expected.name
        # 断言插入后的结果频率与预期结果相同
        assert result.freq == expected.freq
        # 断言插入后的结果时区与预期结果相同
        assert result.tz == expected.tz
    # 定义测试函数，测试在没有频率的情况下插入元素
    def test_insert4_no_freq(self, unit, tz, to_ts):
        # 创建一个日期范围，从 "1/1/2000 09:00" 开始，共6个小时，频率为"h"，时区为tz，名称为"idx"，单元为unit
        idx = date_range(
            "1/1/2000 09:00", periods=6, freq="h", tz=tz, name="idx", unit=unit
        )
        # 创建一个预期的 DatetimeIndex 对象，包含特定的日期时间索引，时区为tz，频率为None，并转换为指定的单元
        expected = DatetimeIndex(
            [
                "2000-01-01 09:00",
                "2000-01-01 10:00",
                "2000-01-01 11:00",
                "2000-01-01 12:00",
                "2000-01-01 13:00",
                "2000-01-01 14:00",
                "2000-01-01 10:00",
            ],
            name="idx",
            tz=tz,
            freq=None,
        ).as_unit(unit)
        # 将频率重置为None
        d = to_ts(Timestamp("2000-01-01 10:00", tz=tz))
        # 在索引的位置6插入时间戳d
        result = idx.insert(6, d)
        # 断言插入后的结果与预期相等
        tm.assert_index_equal(result, expected)
        # 断言结果的名称与预期的名称相等
        assert result.name == expected.name
        # 断言结果的时区与预期的时区相等
        assert result.tz == expected.tz
        # 断言结果的频率为None
        assert result.freq is None

    # TODO: also changes DataFrame.__setitem__ with expansion
    # 定义测试函数，测试在时区不匹配的情况下插入元素
    def test_insert_mismatched_tzawareness(self):
        # 创建一个日期范围，从 "1/1/2000" 开始，共3天，频率为"D"，时区为"Asia/Tokyo"，名称为"idx"
        idx = date_range("1/1/2000", periods=3, freq="D", tz="Asia/Tokyo", name="idx")

        # 创建一个不匹配时区的时间戳
        item = Timestamp("2000-01-04")
        # 在索引的位置3插入时间戳item
        result = idx.insert(3, item)
        # 创建预期的 Index 对象，包含插入后的索引
        expected = Index(
            list(idx[:3]) + [item] + list(idx[3:]), dtype=object, name="idx"
        )
        # 断言插入后的结果与预期相等
        tm.assert_index_equal(result, expected)

        # 创建一个不匹配时区的 datetime 对象
        item = datetime(2000, 1, 4)
        # 在索引的位置3插入datetime对象item
        result = idx.insert(3, item)
        # 创建预期的 Index 对象，包含插入后的索引
        expected = Index(
            list(idx[:3]) + [item] + list(idx[3:]), dtype=object, name="idx"
        )
        # 断言插入后的结果与预期相等
        tm.assert_index_equal(result, expected)

    # TODO: also changes DataFrame.__setitem__ with expansion
    # 定义测试函数，测试在时区不匹配的情况下插入元素
    def test_insert_mismatched_tz(self):
        # 创建一个日期范围，从 "1/1/2000" 开始，共3天，频率为"D"，时区为"Asia/Tokyo"，名称为"idx"
        idx = date_range("1/1/2000", periods=3, freq="D", tz="Asia/Tokyo", name="idx")

        # 创建一个不匹配时区的时间戳对象
        item = Timestamp("2000-01-04", tz="US/Eastern")
        # 在索引的位置3插入时间戳item
        result = idx.insert(3, item)
        # 创建预期的 Index 对象，包含插入后的索引，并将插入项的时区转换为idx的时区
        expected = Index(
            list(idx[:3]) + [item.tz_convert(idx.tz)] + list(idx[3:]),
            name="idx",
        )
        # 断言插入后的结果与预期相等
        assert expected.dtype == idx.dtype
        tm.assert_index_equal(result, expected)

        # 创建一个不匹配时区的 datetime 对象
        item = datetime(2000, 1, 4, tzinfo=zoneinfo.ZoneInfo("US/Eastern"))
        # 在索引的位置3插入datetime对象item
        result = idx.insert(3, item)
        # 创建预期的 Index 对象，包含插入后的索引，并将插入项的时区转换为idx的时区
        expected = Index(
            list(idx[:3]) + [item.astimezone(idx.tzinfo)] + list(idx[3:]),
            name="idx",
        )
        # 断言插入后的结果与预期相等
        assert expected.dtype == idx.dtype
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "item", [0, np.int64(0), np.float64(0), np.array(0), np.timedelta64(456)]
    )
    # 当插入不匹配类型的项时引发异常的测试函数
    def test_insert_mismatched_types_raises(self, tz_aware_fixture, item):
        # 获取时区感知的fixture对象
        tz = tz_aware_fixture
        # 创建一个日期范围，从"2019-11-04"开始，向前9天，频率为每天一次，命名为9，应用给定的时区
        dti = date_range("2019-11-04", periods=9, freq="-1D", name=9, tz=tz)

        # 在索引1处插入item
        result = dti.insert(1, item)

        # 如果item是一个NumPy数组，则断言其单个元素为0
        if isinstance(item, np.ndarray):
            assert item.item() == 0
            # 创建预期的索引对象，包含插入前的第一个元素和0，以及其余的元素
            expected = Index([dti[0], 0] + list(dti[1:]), dtype=object, name=9)
        else:
            # 创建预期的索引对象，包含插入前的第一个元素和item，以及其余的元素
            expected = Index([dti[0], item] + list(dti[1:]), dtype=object, name=9)

        # 断言插入后的结果与预期相等
        tm.assert_index_equal(result, expected)

    # 测试插入可以转换为字符串的值的情况
    def test_insert_castable_str(self, tz_aware_fixture):
        # 获取时区感知的fixture对象
        tz = tz_aware_fixture
        # 创建一个日期范围，从"2019-11-04"开始，向前3天，频率为每天一次，命名为9，应用给定的时区
        dti = date_range("2019-11-04", periods=3, freq="-1D", name=9, tz=tz)

        # 插入值为"2019-11-05"到索引0处
        value = "2019-11-05"
        result = dti.insert(0, value)

        # 创建时间戳对象，将值本地化到给定的时区
        ts = Timestamp(value).tz_localize(tz)
        # 创建预期的时间日期索引对象，包含插入的时间戳和之前的元素
        expected = DatetimeIndex([ts] + list(dti), dtype=dti.dtype, name=9)
        # 断言插入后的结果与预期相等
        tm.assert_index_equal(result, expected)

    # 测试插入不可转换为字符串的值的情况
    def test_insert_non_castable_str(self, tz_aware_fixture):
        # 获取时区感知的fixture对象
        tz = tz_aware_fixture
        # 创建一个日期范围，从"2019-11-04"开始，向前3天，频率为每天一次，命名为9，应用给定的时区
        dti = date_range("2019-11-04", periods=3, freq="-1D", name=9, tz=tz)

        # 插入值为"foo"到索引0处
        value = "foo"
        result = dti.insert(0, value)

        # 创建预期的索引对象，包含插入的字符串和之前的元素
        expected = Index(["foo"] + list(dti), dtype=object, name=9)
        # 断言插入后的结果与预期相等
        tm.assert_index_equal(result, expected)
```