# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\test_scalar_compat.py`

```
"""
Tests for TimedeltaIndex methods behaving like their Timedelta counterparts
"""

import numpy as np
import pytest

from pandas._libs.tslibs.offsets import INVALID_FREQ_ERR_MSG

from pandas import (
    Index,
    Series,
    Timedelta,
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm


class TestVectorizedTimedelta:
    def test_tdi_total_seconds(self):
        # GH#10939
        # test index
        # 创建一个时间增量范围，从指定时间开始，每秒增加，共两个时间点
        rng = timedelta_range("1 days, 10:11:12.100123456", periods=2, freq="s")
        # 预期结果，将时间增量转换为总秒数
        expt = [
            1 * 86400 + 10 * 3600 + 11 * 60 + 12 + 100123456.0 / 1e9,
            1 * 86400 + 10 * 3600 + 11 * 60 + 13 + 100123456.0 / 1e9,
        ]
        # 断言结果几乎等于预期的索引
        tm.assert_almost_equal(rng.total_seconds(), Index(expt))

        # test Series
        # 将时间增量范围转换为 Series 对象
        ser = Series(rng)
        # 预期的 Series 结果
        s_expt = Series(expt, index=[0, 1])
        # 断言 Series 对象的 total_seconds 方法结果与预期结果相等
        tm.assert_series_equal(ser.dt.total_seconds(), s_expt)

        # with nat
        # 设置其中一个值为 NaN
        ser[1] = np.nan
        # 更新预期的 Series 结果，包含 NaN 值
        s_expt = Series(
            [1 * 86400 + 10 * 3600 + 11 * 60 + 12 + 100123456.0 / 1e9, np.nan],
            index=[0, 1],
        )
        # 断言 Series 对象的 total_seconds 方法结果与更新后的预期结果相等
        tm.assert_series_equal(ser.dt.total_seconds(), s_expt)

    def test_tdi_total_seconds_all_nat(self):
        # with both nat
        # 创建一个全为 NaN 的 Series，数据类型为 timedelta64[ns]
        ser = Series([np.nan, np.nan], dtype="timedelta64[ns]")
        # 对该 Series 调用 dt.total_seconds 方法
        result = ser.dt.total_seconds()
        # 预期的结果，全为 NaN
        expected = Series([np.nan, np.nan])
        # 断言结果与预期相等
        tm.assert_series_equal(result, expected)

    def test_tdi_round(self):
        # 创建一个时间增量范围，从指定时间开始，每 30 分钟增加，共五个时间点
        td = timedelta_range(start="16801 days", periods=5, freq="30Min")
        # 获取其中的第二个时间点
        elt = td[1]

        # 预期的时间增量索引，按小时舍入
        expected_rng = TimedeltaIndex(
            [
                Timedelta("16801 days 00:00:00"),
                Timedelta("16801 days 00:00:00"),
                Timedelta("16801 days 01:00:00"),
                Timedelta("16801 days 02:00:00"),
                Timedelta("16801 days 02:00:00"),
            ]
        )
        # 预期的第二个时间点，按小时舍入
        expected_elt = expected_rng[1]

        # 断言时间增量范围对象按指定频率舍入后与预期结果相等
        tm.assert_index_equal(td.round(freq="h"), expected_rng)
        # 断言单个时间增量对象按指定频率舍入后与预期结果相等
        assert elt.round(freq="h") == expected_elt

        # 预期的错误消息
        msg = INVALID_FREQ_ERR_MSG
        # 使用 pytest 检查舍入到无效频率时是否引发 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            td.round(freq="foo")
        with pytest.raises(ValueError, match=msg):
            elt.round(freq="foo")

        # 预期的错误消息
        msg = "<MonthEnd> is a non-fixed frequency"
        # 使用 pytest 检查舍入到非固定频率时是否引发 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            td.round(freq="ME")
        with pytest.raises(ValueError, match=msg):
            elt.round(freq="ME")

    @pytest.mark.parametrize(
        "freq,msg",
        [
            ("YE", "<YearEnd: month=12> is a non-fixed frequency"),
            ("ME", "<MonthEnd> is a non-fixed frequency"),
            ("foobar", "Invalid frequency: foobar"),
        ],
    )
    # 定义测试函数，用于测试在给定频率下舍入操作的异常情况
    def test_tdi_round_invalid(self, freq, msg):
        # 创建一个时间跨度范围对象，包含3个时间点，频率为"1 min 2 s 3 us"
        t1 = timedelta_range("1 days", periods=3, freq="1 min 2 s 3 us")

        # 使用 pytest 来验证是否会抛出 ValueError 异常，异常消息匹配 msg
        with pytest.raises(ValueError, match=msg):
            t1.round(freq)
        with pytest.raises(ValueError, match=msg):
            # 对 TimedeltaArray 进行相同的舍入测试
            t1._data.round(freq)

    # TODO: 与 test_tdi_round 进行功能去重
    # 定义测试函数，用于测试时间跨度对象的舍入功能
    def test_round(self):
        # 创建一个时间跨度范围对象，包含3个时间点，频率为"1 min 2 s 3 us"
        t1 = timedelta_range("1 days", periods=3, freq="1 min 2 s 3 us")
        # 创建一个负数时间跨度范围对象
        t2 = -1 * t1
        # 创建一个不包含微秒的时间跨度范围对象
        t1a = timedelta_range("1 days", periods=3, freq="1 min 2 s")
        # 将一天转换为纳秒单位的 TimedeltaIndex 对象
        t1c = TimedeltaIndex(np.array([1, 1, 1], "m8[D]")).as_unit("ns")

        # 注意：负数时间向下舍入！因此不会给出整数结果
        # 迭代测试不同频率下的舍入操作结果
        for freq, s1, s2 in [
            ("ns", t1, t2),
            ("us", t1, t2),
            (
                "ms",
                t1a,
                TimedeltaIndex(
                    ["-1 days +00:00:00", "-2 days +23:58:58", "-2 days +23:57:56"]
                ),
            ),
            (
                "s",
                t1a,
                TimedeltaIndex(
                    ["-1 days +00:00:00", "-2 days +23:58:58", "-2 days +23:57:56"]
                ),
            ),
            ("12min", t1c, TimedeltaIndex(["-1 days", "-1 days", "-1 days"])),
            ("h", t1c, TimedeltaIndex(["-1 days", "-1 days", "-1 days"])),
            ("d", t1c, -1 * t1c),
        ]:
            # 对 t1 执行频率 freq 的舍入操作，与预期结果 s1 进行断言比较
            r1 = t1.round(freq)
            tm.assert_index_equal(r1, s1)
            # 对 t2 执行频率 freq 的舍入操作，与预期结果 s2 进行断言比较
            r2 = t2.round(freq)
            tm.assert_index_equal(r2, s2)

    # 定义测试函数，用于测试时间跨度范围对象的组件提取功能
    def test_components(self):
        # 创建一个时间跨度范围对象，起始时间为 "1 days, 10:11:12"，包含2个时间点，频率为 "s"
        rng = timedelta_range("1 days, 10:11:12", periods=2, freq="s")
        # 提取时间跨度范围对象的各个时间组件

        # 创建一个 Series 对象，将 rng 作为其数据
        s = Series(rng)
        # 将 Series 的第二个元素设为 NaN
        s[1] = np.nan

        # 提取 Series 的 dt 属性的 components，返回结果
        result = s.dt.components
        # 断言结果的第一个元素不全为 NaN
        assert not result.iloc[0].isna().all()
        # 断言结果的第二个元素全为 NaN
        assert result.iloc[1].isna().all()
```