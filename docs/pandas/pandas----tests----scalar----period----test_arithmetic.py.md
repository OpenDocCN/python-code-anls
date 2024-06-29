# `D:\src\scipysrc\pandas\pandas\tests\scalar\period\test_arithmetic.py`

```
    # 从 datetime 模块导入 timedelta 类
    from datetime import timedelta

    # 导入 numpy 库并用 np 别名引用
    import numpy as np

    # 导入 pytest 库
    import pytest

    # 从 pandas._libs.tslibs.period 模块导入 IncompatibleFrequency 异常类
    from pandas._libs.tslibs.period import IncompatibleFrequency

    # 从 pandas 模块中导入多个对象，包括 NaT, Period, Timedelta, Timestamp, offsets
    from pandas import (
        NaT,
        Period,
        Timedelta,
        Timestamp,
        offsets,
    )


    # 定义 TestPeriodArithmetic 类
    class TestPeriodArithmetic:
        
        # 定义 test_add_overflow_raises 方法
        def test_add_overflow_raises(self):
            # GH#55503
            # 获取 Timestamp 的最大值，并转换为 "ns" 频率的 Period 对象
            per = Timestamp.max.to_period("ns")

            # 定义错误消息字符串，包括两种可能的错误信息
            msg = "|".join(
                [
                    "Python int too large to convert to C long",
                    # windows, 32bit linux builds
                    "int too big to convert",
                ]
            )

            # 使用 pytest 的 raises 方法检查是否引发 OverflowError 异常，并匹配特定的错误消息
            with pytest.raises(OverflowError, match=msg):
                per + 1

            # 定义第二个错误消息字符串
            msg = "value too large"

            # 进行两次异常检查，检查是否引发 OverflowError 异常，并匹配特定的错误消息
            with pytest.raises(OverflowError, match=msg):
                per + Timedelta(1)
            with pytest.raises(OverflowError, match=msg):
                per + offsets.Nano(1)

        # 定义 test_period_add_integer 方法
        def test_period_add_integer(self):
            # 创建两个日期周期对象，频率为 "D"，分别表示 2008 年 1 月 1 日和 2008 年 1 月 2 日
            per1 = Period(freq="D", year=2008, month=1, day=1)
            per2 = Period(freq="D", year=2008, month=1, day=2)

            # 断言表达式，验证周期对象加上整数 1 后是否等于另一个周期对象
            assert per1 + 1 == per2
            assert 1 + per1 == per2

        # 定义 test_period_add_invalid 方法
        def test_period_add_invalid(self):
            # GH#4731
            # 创建两个日期周期对象，频率为 "D"，分别表示 2008 年 1 月 1 日和 2008 年 1 月 2 日
            per1 = Period(freq="D", year=2008, month=1, day=1)
            per2 = Period(freq="D", year=2008, month=1, day=2)

            # 定义错误消息字符串，包括多种可能的错误信息
            msg = "|".join(
                [
                    r"unsupported operand type\(s\)",
                    "can only concatenate str",
                    "must be str, not Period",
                ]
            )

            # 使用 pytest 的 raises 方法检查是否引发 TypeError 异常，并匹配特定的错误消息
            with pytest.raises(TypeError, match=msg):
                per1 + "str"
            with pytest.raises(TypeError, match=msg):
                "str" + per1
            with pytest.raises(TypeError, match=msg):
                per1 + per2

        # 定义 test_period_sub_period_annual 方法
        def test_period_sub_period_annual(self):
            # 创建两个周期对象，分别表示 2011 年和 2007 年
            left, right = Period("2011", freq="Y"), Period("2007", freq="Y")

            # 计算两个周期对象的差值
            result = left - right

            # 断言表达式，验证计算结果是否等于 4 倍 right 的频率
            assert result == 4 * right.freq

            # 定义错误消息字符串，用于验证 IncompatibleFrequency 异常是否会被引发
            msg = r"Input has different freq=M from Period\(freq=Y-DEC\)"

            # 使用 pytest 的 raises 方法检查是否引发 IncompatibleFrequency 异常，并匹配特定的错误消息
            with pytest.raises(IncompatibleFrequency, match=msg):
                left - Period("2007-01", freq="M")

        # 定义 test_period_sub_period 方法
        def test_period_sub_period(self):
            # 创建两个日期周期对象，频率为 "D"，分别表示 2011 年 1 月 1 日和 2011 年 1 月 15 日
            per1 = Period("2011-01-01", freq="D")
            per2 = Period("2011-01-15", freq="D")

            # 获取周期对象的频率
            off = per1.freq

            # 断言表达式，验证两个周期对象相减的结果是否等于 -14 倍频率的 off
            assert per1 - per2 == -14 * off
            assert per2 - per1 == 14 * off

            # 定义错误消息字符串，用于验证 IncompatibleFrequency 异常是否会被引发
            msg = r"Input has different freq=M from Period\(freq=D\)"

            # 使用 pytest 的 raises 方法检查是否引发 IncompatibleFrequency 异常，并匹配特定的错误消息
            with pytest.raises(IncompatibleFrequency, match=msg):
                per1 - Period("2011-02", freq="M")

        # 使用 pytest 的 parametrize 标记定义多个参数化测试用例
        @pytest.mark.parametrize("n", [1, 2, 3, 4])
        def test_sub_n_gt_1_ticks(self, tick_classes, n):
            # GH#23878
            # 创建两个日期周期对象，频率由 tick_classes(n) 参数决定
            p1 = Period("19910905", freq=tick_classes(n))
            p2 = Period("19920406", freq=tick_classes(n))

            # 计算两个周期对象相减的预期结果
            expected = Period(str(p2), freq=p2.freq.base) - Period(
                str(p1), freq=p1.freq.base
            )

            # 断言表达式，验证实际计算结果是否等于预期结果
            assert (p2 - p1) == expected

        # 使用 pytest 的 parametrize 标记定义多个参数化测试用例
        @pytest.mark.parametrize("normalize", [True, False])
        @pytest.mark.parametrize("n", [1, 2, 3, 4])
    @pytest.mark.parametrize(
        "offset, kwd_name",
        [
            (offsets.YearEnd, "month"),  # 参数化测试：使用YearEnd偏移和'month'关键字名称
            (offsets.QuarterEnd, "startingMonth"),  # 使用QuarterEnd偏移和'startingMonth'关键字名称
            (offsets.MonthEnd, None),  # 使用MonthEnd偏移，无关键字名称
            (offsets.Week, "weekday"),  # 使用Week偏移和'weekday'关键字名称
        ],
    )
    def test_sub_n_gt_1_offsets(self, offset, kwd_name, n, normalize):
        # GH#23878
        kwds = {kwd_name: 3} if kwd_name is not None else {}  # 根据关键字名称是否为None创建关键字字典kwds
        p1_d = "19910905"  # 第一个日期字符串
        p2_d = "19920406"  # 第二个日期字符串
        p1 = Period(p1_d, freq=offset(n, normalize, **kwds))  # 创建第一个Period对象
        p2 = Period(p2_d, freq=offset(n, normalize, **kwds))  # 创建第二个Period对象

        expected = Period(p2_d, freq=p2.freq.base) - Period(p1_d, freq=p1.freq.base)  # 计算预期的时间差

        assert (p2 - p1) == expected  # 断言两个Period对象的差等于预期的时间差

    @pytest.mark.parametrize("freq", ["M", "2M", "3M"])
    def test_period_addsub_nat(self, freq):
        # GH#13071
        per = Period("2011-01", freq=freq)  # 创建指定频率的Period对象

        # For subtraction, NaT is treated as another Period object
        assert NaT - per is NaT  # 针对减法操作，NaT被视为另一个Period对象
        assert per - NaT is NaT  # 针对减法操作，Period对象减去NaT结果为NaT

        # For addition, NaT is treated as offset-like
        assert NaT + per is NaT  # 针对加法操作，NaT被视为类似偏移量
        assert per + NaT is NaT  # 针对加法操作，Period对象加上NaT结果为NaT

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "m"])
    def test_period_add_sub_td64_nat(self, unit):
        # GH#47196
        per = Period("2022-06-01", "D")  # 创建指定频率的Period对象
        nat = np.timedelta64("NaT", unit)  # 创建一个NaT类型的numpy timedelta64对象

        assert per + nat is NaT  # Period对象加上NaT结果为NaT
        assert nat + per is NaT  # NaT加上Period对象结果为NaT
        assert per - nat is NaT  # Period对象减去NaT结果为NaT

        with pytest.raises(TypeError, match="unsupported operand"):
            nat - per  # 检查不支持的操作数类型错误异常

    def test_period_ops_offset(self):
        per = Period("2011-04-01", freq="D")  # 创建指定频率的Period对象
        result = per + offsets.Day()  # 使用Day偏移对象进行加法操作
        exp = Period("2011-04-02", freq="D")  # 预期的Period对象

        assert result == exp  # 断言结果与预期相等

        result = per - offsets.Day(2)  # 使用Day(2)偏移对象进行减法操作
        exp = Period("2011-03-30", freq="D")  # 预期的Period对象

        assert result == exp  # 断言结果与预期相等

        msg = r"Input cannot be converted to Period\(freq=D\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            per + offsets.Hour(2)  # 检查不兼容频率的错误异常

        with pytest.raises(IncompatibleFrequency, match=msg):
            per - offsets.Hour(2)  # 检查不兼容频率的错误异常

    def test_period_add_timestamp_raises(self):
        # GH#17983
        ts = Timestamp("2017")  # 创建一个Timestamp对象
        per = Period("2017", freq="M")  # 创建一个指定频率的Period对象

        msg = r"unsupported operand type\(s\) for \+: 'Timestamp' and 'Period'"
        with pytest.raises(TypeError, match=msg):
            ts + per  # 检查不支持的操作数类型错误异常

        msg = r"unsupported operand type\(s\) for \+: 'Period' and 'Timestamp'"
        with pytest.raises(TypeError, match=msg):
            per + ts  # 检查不支持的操作数类型错误异常
class TestPeriodComparisons:
    # 测试相同频率的周期比较
    def test_period_comparison_same_freq(self):
        # 创建两个月度周期对象
        jan = Period("2000-01", "M")
        feb = Period("2000-02", "M")

        # 断言不相等
        assert not jan == feb
        # 断言不相等
        assert jan != feb
        # 断言 jan 小于 feb
        assert jan < feb
        # 断言 jan 小于等于 feb
        assert jan <= feb
        # 断言 jan 不大于 feb
        assert not jan > feb
        # 断言 jan 不大于等于 feb
        assert not jan >= feb

    # 测试相同周期但不同对象的比较
    def test_period_comparison_same_period_different_object(self):
        # 创建两个相同周期的 Period 对象
        left = Period("2000-01", "M")
        right = Period("2000-01", "M")

        # 断言相等
        assert left == right
        # 断言左边大于等于右边
        assert left >= right
        # 断言左边小于等于右边
        assert left <= right
        # 断言左边不小于右边
        assert not left < right
        # 断言左边不大于右边
        assert not left > right

    # 测试不同频率的周期比较
    def test_period_comparison_mismatched_freq(self):
        # 创建一个月度周期和一天的周期对象
        jan = Period("2000-01", "M")
        day = Period("2012-01-01", "D")

        # 断言不相等
        assert not jan == day
        # 断言不相等
        assert jan != day
        # 使用 pytest 检查是否会引发 IncompatibleFrequency 异常
        msg = r"Input has different freq=D from Period\(freq=M\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            jan < day
        with pytest.raises(IncompatibleFrequency, match=msg):
            jan <= day
        with pytest.raises(IncompatibleFrequency, match=msg):
            jan > day
        with pytest.raises(IncompatibleFrequency, match=msg):
            jan >= day

    # 测试与无效类型的周期比较
    def test_period_comparison_invalid_type(self):
        # 创建一个月度周期对象
        jan = Period("2000-01", "M")

        # 断言不相等
        assert not jan == 1
        # 断言不相等
        assert jan != 1

        # 准备错误消息的一部分，指明不支持的操作类型
        int_or_per = "'(Period|int)'"
        msg = f"not supported between instances of {int_or_per} and {int_or_per}"
        # 针对不同的类型组合，使用 pytest 检查是否会引发 TypeError 异常
        for left, right in [(jan, 1), (1, jan)]:
            with pytest.raises(TypeError, match=msg):
                left > right
            with pytest.raises(TypeError, match=msg):
                left >= right
            with pytest.raises(TypeError, match=msg):
                left < right
            with pytest.raises(TypeError, match=msg):
                left <= right

    # 测试周期与 NaT（Not a Time）的比较
    def test_period_comparison_nat(self):
        # 创建一个每日频率的周期对象
        per = Period("2011-01-01", freq="D")

        # 创建一个时间戳对象
        ts = Timestamp("2011-01-01")
        # 确认 Period('NaT') 与 Timestamp('NaT') 的行为一致
        for left, right in [
            (NaT, per),
            (per, NaT),
            (NaT, ts),
            (ts, NaT),
        ]:
            # 断言左边不小于右边
            assert not left < right
            # 断言左边不大于右边
            assert not left > right
            # 断言左边不等于右边
            assert not left == right
            # 断言左边不等于右边
            assert left != right
            # 断言左边不小于等于右边
            assert not left <= right
            # 断言左边不大于等于右边
            assert not left >= right

    # 使用 pytest 参数化装饰器，测试周期与 NumPy 零维数组的比较
    @pytest.mark.parametrize(
        "scalar, expected",
        ((0, False), (Period("2000-01", "M"), True)),
    )
    def test_period_comparison_numpy_zerodim_arr(self, scalar, expected):
        # 创建一个零维数组
        zerodim_arr = np.array(scalar)
        # 创建一个月度周期对象
        per = Period("2000-01", "M")

        # 断言 per 是否等于 zerodim_arr，结果应符合 expected
        assert (per == zerodim_arr) is expected
        # 断言 zerodim_arr 是否等于 per，结果应符合 expected
        assert (zerodim_arr == per) is expected
```