# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\test_constructors.py`

```
# 导入必要的库
import numpy as np
import pytest

# 导入异常类，用于处理频率不兼容情况
from pandas._libs.tslibs.period import IncompatibleFrequency

# 导入周期数据类型
from pandas.core.dtypes.dtypes import PeriodDtype

# 导入 pandas 的核心组件
from pandas import (
    Index,
    NaT,
    Period,
    PeriodIndex,
    Series,
    date_range,
    offsets,
    period_range,
)

# 导入 pandas 的测试工具
import pandas._testing as tm

# 导入周期数组
from pandas.core.arrays import PeriodArray


class TestPeriodIndexDisallowedFreqs:
    @pytest.mark.parametrize(
        "freq,freq_depr",
        [
            ("2M", "2ME"),
            ("2Q-MAR", "2QE-MAR"),
            ("2Y-FEB", "2YE-FEB"),
            ("2M", "2me"),
            ("2Q-MAR", "2qe-MAR"),
            ("2Y-FEB", "2yE-feb"),
        ],
    )
    def test_period_index_offsets_frequency_error_message(self, freq, freq_depr):
        # 测试特定频率下的错误消息
        msg = f"Invalid frequency: {freq_depr}"

        # 确保创建 PeriodIndex 对象时会触发 ValueError 异常，并检查错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2020-01-01", "2020-01-02"], freq=freq_depr)

        # 确保使用 period_range 函数时会触发 ValueError 异常，并检查错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            period_range(start="2020-01-01", end="2020-01-02", freq=freq_depr)

    @pytest.mark.parametrize(
        "freq",
        ["2SME", "2sme", "2BYE", "2Bye", "2CBME"],
    )
    def test_period_index_frequency_invalid_freq(self, freq):
        # 测试无效频率时的错误消息
        msg = f"Invalid frequency: {freq}"

        # 确保使用 period_range 函数时会触发 ValueError 异常，并检查错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            period_range("2020-01", "2020-05", freq=freq)

        # 确保创建 PeriodIndex 对象时会触发 ValueError 异常，并检查错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2020-01", "2020-05"], freq=freq)

    @pytest.mark.parametrize("freq", ["2BQE-SEP", "2BYE-MAR", "2BME"])
    def test_period_index_from_datetime_index_invalid_freq(self, freq):
        # 测试从日期时间索引转换为周期索引时的错误消息
        msg = f"Invalid frequency: {freq}"

        # 创建日期范围对象，使用无效频率时应触发 ValueError 异常，并检查错误消息是否匹配
        rng = date_range("01-Jan-2012", periods=8, freq=freq)
        with pytest.raises(ValueError, match=msg):
            rng.to_period()

    @pytest.mark.parametrize("freq_depr", ["2T", "1l", "2U", "n"])
    def test_period_index_T_L_U_N_raises(self, freq_depr):
        # 测试特定频率缩写时的错误消息
        msg = f"Invalid frequency: {freq_depr}"

        # 确保使用 period_range 函数时会触发 ValueError 异常，并检查错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            period_range("2020-01", "2020-05", freq=freq_depr)

        # 确保创建 PeriodIndex 对象时会触发 ValueError 异常，并检查错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2020-01", "2020-05"], freq=freq_depr)


class TestPeriodIndex:
    def test_from_ordinals(self):
        # 测试从序数创建周期对象
        Period(ordinal=-1000, freq="Y")
        Period(ordinal=0, freq="Y")

        # 使用 from_ordinals 方法创建周期索引，验证两种不同输入方式的结果相等
        idx1 = PeriodIndex.from_ordinals(ordinals=[-1, 0, 1], freq="Y")
        idx2 = PeriodIndex.from_ordinals(ordinals=np.array([-1, 0, 1]), freq="Y")
        tm.assert_index_equal(idx1, idx2)
    def test_construction_base_constructor(self):
        # GH 13664
        # 创建一个包含 Period 对象、NaT 和另一个 Period 对象的数组
        arr = [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="M")]
        # 测试通过创建 Index 对象和 PeriodIndex 对象来验证它们的相等性
        tm.assert_index_equal(Index(arr), PeriodIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), PeriodIndex(np.array(arr)))

        # 创建一个包含 np.nan、NaT 和 Period 对象的数组
        arr = [np.nan, NaT, Period("2011-03", freq="M")]
        # 测试通过创建 Index 对象和 PeriodIndex 对象来验证它们的相等性
        tm.assert_index_equal(Index(arr), PeriodIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), PeriodIndex(np.array(arr)))

        # 创建一个包含 Period 对象、NaT 和另一个具有不同频率的 Period 对象的数组
        arr = [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="D")]
        # 测试通过创建 Index 对象，并使用 dtype=object 参数来验证其相等性
        tm.assert_index_equal(Index(arr), Index(arr, dtype=object))
        tm.assert_index_equal(Index(np.array(arr)), Index(np.array(arr), dtype=object))

    def test_base_constructor_with_period_dtype(self):
        # 使用 PeriodDtype 创建 dtype
        dtype = PeriodDtype("D")
        # 创建一个包含日期字符串的列表
        values = ["2011-01-01", "2012-03-04", "2014-05-01"]
        # 使用 Index 对象来创建一个具有指定 dtype 的索引
        result = Index(values, dtype=dtype)

        # 预期的结果应当是一个 PeriodIndex 对象，使用相同的 dtype
        expected = PeriodIndex(values, dtype=dtype)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "values_constructor", [list, np.array, PeriodIndex, PeriodArray._from_sequence]
    )
    def test_index_object_dtype(self, values_constructor):
        # Index(periods, dtype=object) 是一个 Index 对象（而不是 PeriodIndex）
        # 创建一个包含 Period 对象、NaT 和另一个 Period 对象的列表
        periods = [
            Period("2011-01", freq="M"),
            NaT,
            Period("2011-03", freq="M"),
        ]
        # 根据给定的构造函数创建值
        values = values_constructor(periods)
        # 使用 dtype=object 创建 Index 对象
        result = Index(values, dtype=object)

        # 断言结果的类型应当是 Index
        assert type(result) is Index
        # 使用 tm.assert_numpy_array_equal 断言值数组与期望的 np.array(values) 相等
        tm.assert_numpy_array_equal(result.values, np.array(values))

    def test_constructor_use_start_freq(self):
        # GH #1118
        msg1 = "Period with BDay freq is deprecated"
        # 断言在运行代码块时会产生 FutureWarning 警告，警告信息与 msg1 匹配
        with tm.assert_produces_warning(FutureWarning, match=msg1):
            p = Period("4/2/2012", freq="B")
        msg2 = r"PeriodDtype\[B\] is deprecated"
        # 断言在运行代码块时会产生 FutureWarning 警告，警告信息与 msg2 匹配
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            expected = period_range(start="4/2/2012", periods=10, freq="B")

        # 断言在运行代码块时会产生 FutureWarning 警告，警告信息与 msg2 匹配
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            index = period_range(start=p, periods=10)
        # 使用 tm.assert_index_equal 断言 index 与 expected 相等
        tm.assert_index_equal(index, expected)
    def test_constructor_field_arrays(self):
        # GH #1264
        # 创建年份数组，从1990到2009，每个年份重复4次，然后切片去掉前两个和后两个元素
        years = np.arange(1990, 2010).repeat(4)[2:-2]
        # 创建季度数组，包含1到4，重复20次，然后切片去掉前两个和后两个元素
        quarters = np.tile(np.arange(1, 5), 20)[2:-2]

        # 使用年份和季度数组创建 PeriodIndex 对象，频率为"Q-DEC"
        index = PeriodIndex.from_fields(year=years, quarter=quarters, freq="Q-DEC")
        # 期望的 PeriodIndex 对象，从"1990Q3"到"2009Q2"，频率为"Q-DEC"
        expected = period_range("1990Q3", "2009Q2", freq="Q-DEC")
        # 断言 index 和 expected 相等
        tm.assert_index_equal(index, expected)

        # 使用相同的年份和季度数组创建另一个 PeriodIndex 对象，频率为"2Q-DEC"
        index2 = PeriodIndex.from_fields(year=years, quarter=quarters, freq="2Q-DEC")
        # 断言 index 的 asi8 属性与 index2 的 asi8 属性相等
        tm.assert_numpy_array_equal(index.asi8, index2.asi8)

        # 使用年份和季度数组创建 PeriodIndex 对象，频率为默认的"Q-DEC"
        index = PeriodIndex.from_fields(year=years, quarter=quarters)
        # 断言 index 和 expected 相等
        tm.assert_index_equal(index, expected)

        # 设置新的年份和月份数组
        years = [2007, 2007, 2007]
        months = [1, 2]

        # 预期的错误消息
        msg = "Mismatched Period array lengths"
        # 使用 pytest 的 raises 函数检查 ValueError 异常，匹配错误消息
        with pytest.raises(ValueError, match=msg):
            # 用年份和月份数组创建 PeriodIndex 对象，频率为"M"
            PeriodIndex.from_fields(year=years, month=months, freq="M")
        with pytest.raises(ValueError, match=msg):
            # 用年份和月份数组创建另一个 PeriodIndex 对象，频率为"2M"
            PeriodIndex.from_fields(year=years, month=months, freq="2M")

        # 设置新的年份和月份数组
        years = [2007, 2007, 2007]
        months = [1, 2, 3]
        # 用年份和月份数组创建 PeriodIndex 对象，频率为"M"
        idx = PeriodIndex.from_fields(year=years, month=months, freq="M")
        # 期望的 PeriodIndex 对象，从"2007-01"开始，共3个期间，频率为"M"
        exp = period_range("2007-01", periods=3, freq="M")
        # 断言 idx 和 exp 相等
        tm.assert_index_equal(idx, exp)

    def test_constructor_nano(self):
        # 创建纳秒频率的 PeriodIndex 对象
        idx = period_range(
            start=Period(ordinal=1, freq="ns"),
            end=Period(ordinal=4, freq="ns"),
            freq="ns",
        )
        # 期望的纳秒频率的 PeriodIndex 对象
        exp = PeriodIndex(
            [
                Period(ordinal=1, freq="ns"),
                Period(ordinal=2, freq="ns"),
                Period(ordinal=3, freq="ns"),
                Period(ordinal=4, freq="ns"),
            ],
            freq="ns",
        )
        # 断言 idx 和 exp 相等
        tm.assert_index_equal(idx, exp)

    def test_constructor_arrays_negative_year(self):
        # 创建年份和季度数组
        years = np.arange(1960, 2000, dtype=np.int64).repeat(4)
        quarters = np.tile(np.array([1, 2, 3, 4], dtype=np.int64), 40)
        # 使用年份和季度数组创建 PeriodIndex 对象
        pindex = PeriodIndex.from_fields(year=years, quarter=quarters)

        # 断言 pindex 的 year 属性与 Index(years) 相等
        tm.assert_index_equal(pindex.year, Index(years))
        # 断言 pindex 的 quarter 属性与 Index(quarters) 相等
        tm.assert_index_equal(pindex.quarter, Index(quarters))

    def test_constructor_invalid_quarters(self):
        # 预期的错误消息
        msg = "Quarter must be 1 <= q <= 4"
        # 使用 pytest 的 raises 函数检查 ValueError 异常，匹配错误消息
        with pytest.raises(ValueError, match=msg):
            # 用指定年份和季度范围创建 PeriodIndex 对象，频率为"Q-DEC"
            PeriodIndex.from_fields(
                year=range(2000, 2004), quarter=list(range(4)), freq="Q-DEC"
            )

    def test_period_range_fractional_period(self):
        # 预期的错误消息
        msg = "periods must be an integer, got 10.5"
        # 使用 pytest 的 raises 函数检查 TypeError 异常，匹配错误消息
        with pytest.raises(TypeError, match=msg):
            # 创建一个月频率的 PeriodIndex 对象，期数为10.5，会引发异常
            period_range("2007-01", periods=10.5, freq="M")

    def test_constructor_with_without_freq(self):
        # GH53687
        # 创建一个以"2002-01-01 00:00"为起始时间，频率为"30min"的 Period 对象
        start = Period("2002-01-01 00:00", freq="30min")
        # 创建一个预期的 PeriodIndex 对象，从 start 开始，共5个期间，频率与 start 相同
        exp = period_range(start=start, periods=5, freq=start.freq)
        # 创建一个实际的 PeriodIndex 对象，从 start 开始，共5个期间，频率为默认
        result = period_range(start=start, periods=5)
        # 断言 exp 和 result 相等
        tm.assert_index_equal(exp, result)
    # 定义一个测试方法，用于测试从数组样式构造 PeriodIndex 对象
    def test_constructor_fromarraylike(self):
        # 创建一个 PeriodIndex 对象，包含从 "2007-01" 开始的 20 个月，频率为每月
        idx = period_range("2007-01", periods=20, freq="M")

        # 验证 idx.values 是 Period 对象数组，因此可以检索频率信息
        tm.assert_index_equal(PeriodIndex(idx.values), idx)
        tm.assert_index_equal(PeriodIndex(list(idx.values)), idx)

        # 设置错误消息，用于异常验证
        msg = "freq not specified and cannot be inferred"
        # 验证构造 PeriodIndex 对象时，如果未指定频率则会抛出 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(idx.asi8)
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(list(idx.asi8))

        # 设置错误消息，用于异常验证
        msg = "'Period' object is not iterable"
        # 验证构造 PeriodIndex 对象时，如果传入的数据不可迭代，则会抛出 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            PeriodIndex(data=Period("2007", freq="Y"))

        # 使用迭代器构造 PeriodIndex 对象
        result = PeriodIndex(iter(idx))
        tm.assert_index_equal(result, idx)

        # 直接使用 idx 构造 PeriodIndex 对象
        result = PeriodIndex(idx)
        tm.assert_index_equal(result, idx)

        # 指定频率 "M" 构造 PeriodIndex 对象
        result = PeriodIndex(idx, freq="M")
        tm.assert_index_equal(result, idx)

        # 使用 offsets.MonthEnd() 频率构造 PeriodIndex 对象
        result = PeriodIndex(idx, freq=offsets.MonthEnd())
        tm.assert_index_equal(result, idx)
        assert result.freq == "ME"

        # 指定频率 "2M" 构造 PeriodIndex 对象
        result = PeriodIndex(idx, freq="2M")
        tm.assert_index_equal(result, idx.asfreq("2M"))
        assert result.freq == "2ME"

        # 使用 offsets.MonthEnd(2) 频率构造 PeriodIndex 对象
        result = PeriodIndex(idx, freq=offsets.MonthEnd(2))
        tm.assert_index_equal(result, idx.asfreq("2M"))
        assert result.freq == "2ME"

        # 指定频率 "D" 构造 PeriodIndex 对象
        result = PeriodIndex(idx, freq="D")
        # 将 idx 转换为每日频率的 PeriodIndex 对象，期望结果是 exp
        exp = idx.asfreq("D", "e")
        tm.assert_index_equal(result, exp)

    # 定义一个测试方法，用于测试从 datetime64 数组构造 PeriodIndex 对象
    def test_constructor_datetime64arr(self):
        # 创建一个 datetime64 数组，起始于 100000 微秒，每隔 100 微秒增加，共 10000 个值
        vals = np.arange(100000, 100000 + 10000, 100, dtype=np.int64)
        # 将 vals 视图转换为 "M8[us]" 类型的 datetime64 数组
        vals = vals.view(np.dtype("M8[us]"))

        # 使用频率 "D" 构造 PeriodIndex 对象
        pi = PeriodIndex(vals, freq="D")

        # 将 vals 转换为 "M8[ns]" 类型的 datetime64 数组，然后使用频率 "D" 构造期望的 PeriodIndex 对象
        expected = PeriodIndex(vals.astype("M8[ns]"), freq="D")
        tm.assert_index_equal(pi, expected)

    # 使用不同的 box 参数进行参数化测试构造 PeriodIndex 对象
    @pytest.mark.parametrize("box", [None, "series", "index"])
    def test_constructor_datetime64arr_ok(self, box):
        # 创建一个日期范围，从 "2017" 年开始，包含 4 个月，频率为每月结束
        data = date_range("2017", periods=4, freq="ME")
        if box is None:
            # 如果 box 为 None，则将 data 转换为其内部的 _values 数组
            data = data._values
        elif box == "series":
            # 如果 box 为 "series"，则将 data 转换为 Series 对象
            data = Series(data)

        # 使用频率 "D" 构造 PeriodIndex 对象
        result = PeriodIndex(data, freq="D")
        # 用包含日期字符串的列表构造期望的 PeriodIndex 对象
        expected = PeriodIndex(
            ["2017-01-31", "2017-02-28", "2017-03-31", "2017-04-30"], freq="D"
        )
        tm.assert_index_equal(result, expected)
    def test_constructor_dtype(self):
        # 使用指定的 dtype 创建 PeriodIndex 对象，时区信息会被本地化处理
        idx = PeriodIndex(["2013-01", "2013-03"], dtype="period[M]")
        # 期望的 PeriodIndex 对象，频率设置为每月
        exp = PeriodIndex(["2013-01", "2013-03"], freq="M")
        # 断言 idx 与 exp 相等
        tm.assert_index_equal(idx, exp)
        # 检查 idx 的 dtype 是否为 "period[M]"
        assert idx.dtype == "period[M]"

        # 使用指定的 dtype 创建 PeriodIndex 对象，频率为每 3 天
        idx = PeriodIndex(["2013-01-05", "2013-03-05"], dtype="period[3D]")
        # 期望的 PeriodIndex 对象，频率设置为每 3 天
        exp = PeriodIndex(["2013-01-05", "2013-03-05"], freq="3D")
        # 断言 idx 与 exp 相等
        tm.assert_index_equal(idx, exp)
        # 检查 idx 的 dtype 是否为 "period[3D]"

        # 如果已经存在频率信息且不相同，则不改变频率
        idx = PeriodIndex(["2013-01-01", "2013-01-02"], freq="D")

        # 使用指定的 dtype 创建 PeriodIndex 对象，频率从日改为月
        res = PeriodIndex(idx, dtype="period[M]")
        # 期望的 PeriodIndex 对象，频率设置为每月
        exp = PeriodIndex(["2013-01", "2013-01"], freq="M")
        # 断言 res 与 exp 相等
        tm.assert_index_equal(res, exp)
        # 检查 res 的 dtype 是否为 "period[M]"

        # 使用指定的 freq 创建 PeriodIndex 对象，频率设置为每月
        res = PeriodIndex(idx, freq="M")
        # 断言 res 与 exp 相等
        tm.assert_index_equal(res, exp)
        # 检查 res 的 dtype 是否为 "period[M]"

        # 当指定的频率和 dtype 不匹配时，应引发异常
        msg = "specified freq and dtype are different"
        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex(["2011-01"], freq="M", dtype="period[D]")

    def test_constructor_empty(self):
        # 使用空列表创建 PeriodIndex 对象，指定频率为每月
        idx = PeriodIndex([], freq="M")
        # 断言 idx 是 PeriodIndex 类的实例
        assert isinstance(idx, PeriodIndex)
        # 断言 idx 的长度为 0
        assert len(idx) == 0
        # 断言 idx 的频率为 "ME"
        assert idx.freq == "ME"

        # 当未指定频率时，应引发值错误异常
        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex([])

    def test_constructor_pi_nat(self):
        # 使用包含 Period 对象和 NaT（Not a Time）的列表创建 PeriodIndex 对象，指定频率为每月
        idx = PeriodIndex(
            [Period("2011-01", freq="M"), NaT, Period("2011-01", freq="M")]
        )
        # 期望的 PeriodIndex 对象，频率设置为每月
        exp = PeriodIndex(["2011-01", "NaT", "2011-01"], freq="M")
        # 断言 idx 与 exp 相等
        tm.assert_index_equal(idx, exp)

        # 使用包含 Period 对象和 NaT 的 NumPy 数组创建 PeriodIndex 对象，指定频率为每月
        idx = PeriodIndex(
            np.array([Period("2011-01", freq="M"), NaT, Period("2011-01", freq="M")])
        )
        # 断言 idx 与 exp 相等
        tm.assert_index_equal(idx, exp)

        # 使用包含多个 NaT 和 Period 对象的列表创建 PeriodIndex 对象，指定频率为每月
        idx = PeriodIndex(
            [NaT, NaT, Period("2011-01", freq="M"), Period("2011-01", freq="M")]
        )
        # 期望的 PeriodIndex 对象，频率设置为每月
        exp = PeriodIndex(["NaT", "NaT", "2011-01", "2011-01"], freq="M")
        # 断言 idx 与 exp 相等
        tm.assert_index_equal(idx, exp)

        # 使用包含多个 NaT 和 Period 对象的 NumPy 数组创建 PeriodIndex 对象，指定频率为每月
        idx = PeriodIndex(
            np.array(
                [NaT, NaT, Period("2011-01", freq="M"), Period("2011-01", freq="M")]
            )
        )
        # 断言 idx 与 exp 相等
        tm.assert_index_equal(idx, exp)

        # 使用包含多个 NaT 和字符串日期的列表创建 PeriodIndex 对象，指定频率为每月
        idx = PeriodIndex([NaT, NaT, "2011-01", "2011-01"], freq="M")
        # 断言 idx 与 exp 相等
        tm.assert_index_equal(idx, exp)

        # 当未指定频率时，应引发值错误异常
        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex([NaT, NaT])

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex(np.array([NaT, NaT]))

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex(["NaT", "NaT"])

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex(np.array(["NaT", "NaT"]))
    # 测试构造函数，检查输入与期间索引频率不兼容的情况
    def test_constructor_incompat_freq(self):
        # 错误消息内容，指示输入的频率与期间索引的频率不匹配
        msg = "Input has different freq=D from PeriodIndex\\(freq=M\\)"

        # 测试使用期间对象列表构造期间索引，期间对象列表中包含不同频率的期间对象
        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex([Period("2011-01", freq="M"), NaT, Period("2011-01", freq="D")])

        # 测试使用包含期间对象的 NumPy 数组构造期间索引，数组中包含不同频率的期间对象
        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex(
                np.array(
                    [Period("2011-01", freq="M"), NaT, Period("2011-01", freq="D")]
                )
            )

        # 测试期间索引的构造，期间对象列表的第一个元素是 NaT
        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex([NaT, Period("2011-01", freq="M"), Period("2011-01", freq="D")])

        # 测试使用包含期间对象的 NumPy 数组构造期间索引，数组中第一个元素是 NaT
        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex(
                np.array(
                    [NaT, Period("2011-01", freq="M"), Period("2011-01", freq="D")]
                )
            )

    # 测试构造函数，检查混合类型输入的情况
    def test_constructor_mixed(self):
        # 构造期间索引，包含不同类型的期间对象
        idx = PeriodIndex(["2011-01", NaT, Period("2011-01", freq="M")])
        # 期望结果的期间索引，指定频率为月
        exp = PeriodIndex(["2011-01", "NaT", "2011-01"], freq="M")
        # 检查实际结果与期望结果是否相等
        tm.assert_index_equal(idx, exp)

        # 构造期间索引，包含多个 NaT 值
        idx = PeriodIndex(["NaT", NaT, Period("2011-01", freq="M")])
        # 期望结果的期间索引，指定频率为月
        exp = PeriodIndex(["NaT", "NaT", "2011-01"], freq="M")
        # 检查实际结果与期望结果是否相等
        tm.assert_index_equal(idx, exp)

        # 构造期间索引，包含字符串和期间对象（指定频率为日）
        idx = PeriodIndex([Period("2011-01-01", freq="D"), NaT, "2012-01-01"])
        # 期望结果的期间索引，指定频率为日
        exp = PeriodIndex(["2011-01-01", "NaT", "2012-01-01"], freq="D")
        # 检查实际结果与期望结果是否相等
        tm.assert_index_equal(idx, exp)

    # 使用参数化测试，检查在构造期间索引时传入浮点数的情况
    @pytest.mark.parametrize("floats", [[1.1, 2.1], np.array([1.1, 2.1])])
    def test_constructor_floats(self, floats):
        # 错误消息内容，指示期间索引的构造不允许包含浮点数
        msg = "PeriodIndex does not allow floating point in construction"
        # 检查在构造期间索引时传入浮点数是否抛出 TypeError 异常，且异常消息匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            PeriodIndex(floats)

    # 测试构造函数，检查通过年和季度构造期间索引的情况
    def test_constructor_year_and_quarter(self):
        # 创建包含年份的 Series 对象
        year = Series([2001, 2002, 2003])
        # 计算每个年份对应的季度
        quarter = year - 2000
        # 使用年份和季度信息构造期间索引
        idx = PeriodIndex.from_fields(year=year, quarter=quarter)
        # 构造期望的期间索引，由季度和年份组成的字符串列表
        strs = [f"{t[0]:d}Q{t[1]:d}" for t in zip(quarter, year)]
        # 将字符串列表转换为 Period 对象列表
        lops = list(map(Period, strs))
        # 使用 Period 对象列表构造期间索引
        p = PeriodIndex(lops)
        # 检查实际结果与期望结果是否相等
        tm.assert_index_equal(p, idx)
    # 定义一个测试方法，用于测试构造频率多重性的情况
    def test_constructor_freq_mult(self):
        # GH #7811
        # 创建一个时间周期索引，起始于 "2014-01"，频率为每2个月，共4个周期
        pidx = period_range(start="2014-01", freq="2M", periods=4)
        # 预期结果是一个时间周期索引，包含指定的日期列表，频率为每2个月
        expected = PeriodIndex(["2014-01", "2014-03", "2014-05", "2014-07"], freq="2M")
        # 断言期望结果与实际结果是否相等
        tm.assert_index_equal(pidx, expected)

        # 创建一个时间周期索引，起始于 "2014-01-02"，结束于 "2014-01-15"，频率为每3天
        pidx = period_range(start="2014-01-02", end="2014-01-15", freq="3D")
        # 预期结果是一个时间周期索引，包含指定的日期列表，频率为每3天
        expected = PeriodIndex(
            ["2014-01-02", "2014-01-05", "2014-01-08", "2014-01-11", "2014-01-14"],
            freq="3D",
        )
        # 断言期望结果与实际结果是否相等
        tm.assert_index_equal(pidx, expected)

        # 创建一个时间周期索引，结束于 "2014-01-01 17:00"，频率为每4小时，共3个周期
        pidx = period_range(end="2014-01-01 17:00", freq="4h", periods=3)
        # 预期结果是一个时间周期索引，包含指定的日期列表，频率为每4小时
        expected = PeriodIndex(
            ["2014-01-01 09:00", "2014-01-01 13:00", "2014-01-01 17:00"], freq="4h"
        )
        # 断言期望结果与实际结果是否相等
        tm.assert_index_equal(pidx, expected)

        # 测试异常情况，预期抛出 ValueError 异常，匹配指定的错误消息
        msg = "Frequency must be positive, because it represents span: -1M"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2011-01"], freq="-1M")

        # 测试异常情况，预期抛出 ValueError 异常，匹配指定的错误消息
        msg = "Frequency must be positive, because it represents span: 0M"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2011-01"], freq="0M")

        # 测试异常情况，预期抛出 ValueError 异常，匹配指定的错误消息
        msg = "Frequency must be positive, because it represents span: 0M"
        with pytest.raises(ValueError, match=msg):
            period_range("2011-01", periods=3, freq="0M")

    # 使用 pytest 的参数化装饰器，对频率偏移和周期进行参数化测试
    @pytest.mark.parametrize(
        "freq_offset, freq_period",
        [
            ("YE", "Y"),
            ("ME", "M"),
            ("D", "D"),
            ("min", "min"),
            ("s", "s"),
        ],
    )
    # 使用 pytest 的参数化装饰器，对倍数进行参数化测试
    @pytest.mark.parametrize("mult", [1, 2, 3, 4, 5])
    # 测试构造函数的频率多重性，同时兼容时间日期索引
    def test_constructor_freq_mult_dti_compat(self, mult, freq_offset, freq_period):
        # 创建一个时间周期索引，起始于 "2014-04-01"，频率为多重倍数后的周期，共10个周期
        freqstr_offset = str(mult) + freq_offset
        freqstr_period = str(mult) + freq_period
        pidx = period_range(start="2014-04-01", freq=freqstr_period, periods=10)
        # 使用日期范围创建预期结果，然后转换为指定频率的时间周期索引
        expected = date_range(
            start="2014-04-01", freq=freqstr_offset, periods=10
        ).to_period(freqstr_period)
        # 断言期望结果与实际结果是否相等
        tm.assert_index_equal(pidx, expected)

    # 使用 pytest 的参数化装饰器，对倍数进行参数化测试
    @pytest.mark.parametrize("mult", [1, 2, 3, 4, 5])
    # 测试构造函数的频率多重性，同时兼容月份的时间日期索引
    def test_constructor_freq_mult_dti_compat_month(self, mult):
        # 创建一个时间周期索引，起始于 "2014-04-01"，频率为多重倍数后的月份，共10个周期
        pidx = period_range(start="2014-04-01", freq=f"{mult}M", periods=10)
        # 使用日期范围创建预期结果，然后转换为指定月份频率的时间周期索引
        expected = date_range(
            start="2014-04-01", freq=f"{mult}ME", periods=10
        ).to_period(f"{mult}M")
        # 断言期望结果与实际结果是否相等
        tm.assert_index_equal(pidx, expected)

    # 测试构造函数的组合频率情况
    def test_constructor_freq_combined(self):
        # 遍历组合频率列表，分别测试每一种组合
        for freq in ["1D1h", "1h1D"]:
            # 创建一个时间周期索引，包含指定的日期列表，指定的组合频率
            pidx = PeriodIndex(["2016-01-01", "2016-01-02"], freq=freq)
            # 预期结果是一个时间周期索引，包含指定的日期列表，频率为每25小时
            expected = PeriodIndex(["2016-01-01 00:00", "2016-01-02 00:00"], freq="25h")
        for freq in ["1D1h", "1h1D"]:
            # 创建一个时间周期索引，起始于 "2016-01-01"，共2个周期，指定的组合频率
            pidx = period_range(start="2016-01-01", periods=2, freq=freq)
            # 预期结果是一个时间周期索引，包含指定的日期列表，频率为每25小时
            expected = PeriodIndex(["2016-01-01 00:00", "2016-01-02 01:00"], freq="25h")
            # 断言期望结果与实际结果是否相等
            tm.assert_index_equal(pidx, expected)
    # 测试不同频率下的时间段长度

    # 创建年度频率的时间段范围，并断言其长度为9年
    pi = period_range(freq="Y", start="1/1/2001", end="12/1/2009")
    assert len(pi) == 9

    # 创建季度频率的时间段范围，并断言其长度为36个季度（4季度 * 9年）
    pi = period_range(freq="Q", start="1/1/2001", end="12/1/2009")
    assert len(pi) == 4 * 9

    # 创建月度频率的时间段范围，并断言其长度为108个月（12个月 * 9年）
    pi = period_range(freq="M", start="1/1/2001", end="12/1/2009")
    assert len(pi) == 12 * 9

    # 创建每日频率的时间段范围，并断言其长度为365 * 9 + 2天（包含2009年的额外两天）
    pi = period_range(freq="D", start="1/1/2001", end="12/31/2009")
    assert len(pi) == 365 * 9 + 2

    # 使用工作日频率创建时间段，预期会产生未来警告，并断言其长度为261 * 9个工作日
    msg = "Period with BDay freq is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        pi = period_range(freq="B", start="1/1/2001", end="12/31/2009")
    assert len(pi) == 261 * 9

    # 创建每小时频率的时间段范围，并断言其长度为365 * 24小时
    pi = period_range(freq="h", start="1/1/2001", end="12/31/2001 23:00")
    assert len(pi) == 365 * 24

    # 创建每分钟频率的时间段范围，并断言其长度为24 * 60分钟
    pi = period_range(freq="Min", start="1/1/2001", end="1/1/2001 23:59")
    assert len(pi) == 24 * 60

    # 创建每秒钟频率的时间段范围，并断言其长度为24 * 60 * 60秒
    pi = period_range(freq="s", start="1/1/2001", end="1/1/2001 23:59:59")
    assert len(pi) == 24 * 60 * 60

    # 使用未来警告断言从指定起始时间创建20个时间段，并验证结果
    with tm.assert_produces_warning(FutureWarning, match=msg):
        start = Period("02-Apr-2005", "B")
        i1 = period_range(start=start, periods=20)
    assert len(i1) == 20
    assert i1.freq == start.freq
    assert i1[0] == start

    # 从指定结束时间创建10个时间段，并验证结果
    end_intv = Period("2006-12-31", "W")
    i1 = period_range(end=end_intv, periods=10)
    assert len(i1) == 10
    assert i1.freq == end_intv.freq
    assert i1[-1] == end_intv

    # 使用未来警告断言从指定结束时间创建10个时间段，并验证结果
    msg = "'w' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        end_intv = Period("2006-12-31", "1w")
    i2 = period_range(end=end_intv, periods=10)
    assert len(i1) == len(i2)
    assert (i1 == i2).all()
    assert i1.freq == i2.freq
    # 测试混合频率应该引发异常
    def test_mixed_freq_raises(self):
        # 设置警告消息内容
        msg = "Period with BDay freq is deprecated"
        # 断言在使用FutureWarning时产生警告，并匹配特定消息
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 创建一个包含"B"频率的Period对象
            end_intv = Period("2005-05-01", "B")

        # 创建Period对象列表
        vals = [end_intv, Period("2006-12-31", "W")]
        # 设置错误消息正则表达式
        msg = r"Input has different freq=W-SUN from PeriodIndex\(freq=B\)"
        # 设置过时消息正则表达式
        depr_msg = r"PeriodDtype\[B\] is deprecated"
        # 断言在捕获IncompatibleFrequency异常时，匹配特定消息，并产生FutureWarning警告
        with pytest.raises(IncompatibleFrequency, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                # 将vals列表作为输入创建PeriodIndex对象
                PeriodIndex(vals)
        # 将vals列表转换为numpy数组
        vals = np.array(vals)
        # 断言在捕获IncompatibleFrequency异常时，匹配特定消息，并产生FutureWarning警告
        with pytest.raises(IncompatibleFrequency, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                # 使用numpy数组vals创建PeriodIndex对象
                PeriodIndex(vals)

    # 参数化测试，根据不同的频率重新创建PeriodIndex对象
    @pytest.mark.parametrize(
        "freq", ["M", "Q", "Y", "D", "B", "min", "s", "ms", "us", "ns", "h"]
    )
    # 忽略特定警告消息
    @pytest.mark.filterwarnings(
        r"ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_recreate_from_data(self, freq):
        # 创建原始Period对象列表
        org = period_range(start="2001/04/01", freq=freq, periods=1)
        # 使用给定频率freq和值org.values创建PeriodIndex对象
        idx = PeriodIndex(org.values, freq=freq)
        # 断言idx对象与org对象相等
        tm.assert_index_equal(idx, org)

    # 使用字符串构造函数进行映射测试
    def test_map_with_string_constructor(self):
        # 创建原始数据列表
        raw = [2005, 2007, 2009]
        # 使用"Y"频率创建PeriodIndex对象
        index = PeriodIndex(raw, freq="Y")

        # 创建预期的Index对象，其中每个元素都是raw中对应元素的字符串形式
        expected = Index([str(num) for num in raw])
        # 使用map函数将index对象中的每个元素转换为字符串
        res = index.map(str)

        # 断言res是一个Index对象
        assert isinstance(res, Index)

        # 断言res中的每个元素都是字符串类型
        assert all(isinstance(resi, str) for resi in res)

        # 最后断言res和expected相等
        tm.assert_index_equal(res, expected)
class TestSimpleNew:
    # 测试简单新建的构造函数
    def test_constructor_simple_new(self):
        # 创建一个时间段范围索引对象
        idx = period_range("2007-01", name="p", periods=2, freq="M")

        # 断言会抛出 AssertionError，匹配特定的错误信息
        with pytest.raises(AssertionError, match="<class .*PeriodIndex'>"):
            idx._simple_new(idx, name="p")

        # 使用 _simple_new 方法创建新对象并断言索引相等
        result = idx._simple_new(idx._data, name="p")
        tm.assert_index_equal(result, idx)

        # 准备错误信息字符串
        msg = "Should be numpy array of type i8"
        # 断言会抛出 AssertionError，匹配特定的错误信息
        with pytest.raises(AssertionError, match=msg):
            # 需要 ndarray，而不是 int64 索引
            type(idx._data)._simple_new(Index(idx.asi8), dtype=idx.dtype)

        # 使用 _simple_new 方法创建新数组对象并断言索引相等
        arr = type(idx._data)._simple_new(idx.asi8, dtype=idx.dtype)
        result = idx._simple_new(arr, name="p")
        tm.assert_index_equal(result, idx)

    # 测试空的简单新建的构造函数
    def test_constructor_simple_new_empty(self):
        # GH13079
        # 创建一个空的时间段索引对象
        idx = PeriodIndex([], freq="M", name="p")
        # 断言会抛出 AssertionError，匹配特定的错误信息
        with pytest.raises(AssertionError, match="<class .*PeriodIndex'>"):
            idx._simple_new(idx, name="p")

        # 使用 _simple_new 方法创建新对象并断言索引相等
        result = idx._simple_new(idx._data, name="p")
        tm.assert_index_equal(result, idx)

    # 参数化测试，检验期间索引简单新建不允许使用浮点数
    @pytest.mark.parametrize("floats", [[1.1, 2.1], np.array([1.1, 2.1])])
    def test_period_index_simple_new_disallows_floats(self, floats):
        # 断言会抛出 AssertionError，匹配特定的错误信息
        with pytest.raises(AssertionError, match="<class "):
            PeriodIndex._simple_new(floats)


class TestShallowCopy:
    # 测试浅复制空对象
    def test_shallow_copy_empty(self):
        # GH#13067
        # 创建一个空的时间段索引对象
        idx = PeriodIndex([], freq="M")
        # 执行浅复制操作
        result = idx._view()
        expected = idx

        # 断言索引对象相等
        tm.assert_index_equal(result, expected)

    # 测试不允许使用 int64 类型进行浅复制
    def test_shallow_copy_disallow_i8(self):
        # GH#24391
        # 创建一个包含三个周期的时间段索引对象
        pi = period_range("2018-01-01", periods=3, freq="2D")
        # 断言会抛出 AssertionError，匹配特定的错误信息
        with pytest.raises(AssertionError, match="ndarray"):
            pi._shallow_copy(pi.asi8)

    # 测试需要禁止使用 PeriodIndex 进行浅复制
    def test_shallow_copy_requires_disallow_period_index(self):
        # 创建一个包含三个周期的时间段索引对象
        pi = period_range("2018-01-01", periods=3, freq="2D")
        # 断言会抛出 AssertionError，匹配特定的错误信息
        with pytest.raises(AssertionError, match="PeriodIndex"):
            pi._shallow_copy(pi)


class TestSeriesPeriod:
    # 测试无法将 PeriodIndex 转换为 float64 类型
    def test_constructor_cant_cast_period(self):
        msg = "Cannot cast PeriodIndex to dtype float64"
        # 断言会抛出 TypeError，匹配特定的错误信息
        with pytest.raises(TypeError, match=msg):
            # 创建一个时间序列对象，尝试将 PeriodIndex 转换为 float64 类型
            Series(period_range("2000-01-01", periods=10, freq="D"), dtype=float)

    # 测试可以将对象转换为对象类型
    def test_constructor_cast_object(self):
        # 创建一个日期范围对象
        pi = period_range("1/1/2000", periods=10)
        # 创建一个时间序列对象，指定 PeriodDtype 类型
        ser = Series(pi, dtype=PeriodDtype("D"))
        # 创建一个期望的时间序列对象
        exp = Series(pi)
        # 断言时间序列对象相等
        tm.assert_series_equal(ser, exp)
```