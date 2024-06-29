# `D:\src\scipysrc\pandas\pandas\tests\scalar\timedelta\methods\test_round.py`

```
# 导入必要的库和模块
from hypothesis import (
    given,
    strategies as st,
)
import numpy as np
import pytest

# 导入 pandas 相关模块和类
from pandas._libs import lib
from pandas._libs.tslibs import iNaT
from pandas.errors import OutOfBoundsTimedelta
from pandas import Timedelta

# 定义 TimedeltaRound 测试类
class TestTimedeltaRound:
    
    # 参数化测试用例，测试 Timedelta 对象的 round 方法
    @pytest.mark.parametrize(
        "freq,s1,s2",
        [
            # 第一个测试用例，频率为 "ns"，s1 和 s2 与 t1、t2 相同
            (
                "ns",
                "1 days 02:34:56.789123456",
                "-1 days 02:34:56.789123456",
            ),
            # 后续测试用例省略
            (
                "us",
                "1 days 02:34:56.789123000",
                "-1 days 02:34:56.789123000",
            ),
            (
                "ms",
                "1 days 02:34:56.789000000",
                "-1 days 02:34:56.789000000",
            ),
            ("s", "1 days 02:34:57", "-1 days 02:34:57"),
            ("2s", "1 days 02:34:56", "-1 days 02:34:56"),
            ("5s", "1 days 02:34:55", "-1 days 02:34:55"),
            ("min", "1 days 02:35:00", "-1 days 02:35:00"),
            ("12min", "1 days 02:36:00", "-1 days 02:36:00"),
            ("h", "1 days 03:00:00", "-1 days 03:00:00"),
            ("d", "1 days", "-1 days"),
        ],
    )
    # 定义测试方法 test_round
    def test_round(self, freq, s1, s2):
        # 将字符串转换为 Timedelta 对象
        s1 = Timedelta(s1)
        s2 = Timedelta(s2)
        t1 = Timedelta("1 days 02:34:56.789123456")
        t2 = Timedelta("-1 days 02:34:56.789123456")

        # 调用 Timedelta 对象的 round 方法
        r1 = t1.round(freq)
        # 断言结果与预期相符
        assert r1 == s1
        r2 = t2.round(freq)
        assert r2 == s2

    # 测试 round 方法处理无效频率时是否引发 ValueError 异常
    def test_round_invalid(self):
        t1 = Timedelta("1 days 02:34:56.789123456")

        for freq, msg in [
            ("YE", "<YearEnd: month=12> is a non-fixed frequency"),
            ("ME", "<MonthEnd> is a non-fixed frequency"),
            ("foobar", "Invalid frequency: foobar"),
        ]:
            with pytest.raises(ValueError, match=msg):
                t1.round(freq)

    # 测试 round 方法边界情况下的行为，主要检查是否引发 OutOfBoundsTimedelta 异常
    @pytest.mark.skip_ubsan
    def test_round_implementation_bounds(self):
        # 检查 Timedelta.min 的 ceil 行为
        result = Timedelta.min.ceil("s")
        expected = Timedelta.min + Timedelta(seconds=1) - Timedelta(145224193)
        assert result == expected

        # 检查 Timedelta.max 的 floor 行为
        result = Timedelta.max.floor("s")
        expected = Timedelta.max - Timedelta(854775807)
        assert result == expected

        # 检查对超出边界的 Timedelta.min 和 Timedelta.max 调用 round 和 floor 是否引发异常
        msg = (
            r"Cannot round -106752 days \+00:12:43.145224193 to freq=s without overflow"
        )
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta.min.floor("s")
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta.min.round("s")

        msg = "Cannot round 106751 days 23:47:16.854775807 to freq=s without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta.max.ceil("s")
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta.max.round("s")
    # 使用 hypothesis 的 given 装饰器，生成一个在指定范围内的整数假设
    @given(val=st.integers(min_value=iNaT + 1, max_value=lib.i8max))
    # 使用 pytest 的 mark.parametrize 装饰器，参数化 method 变量为 Timedelta 类的 round、floor 和 ceil 方法
    @pytest.mark.parametrize(
        "method", [Timedelta.round, Timedelta.floor, Timedelta.ceil]
    )
    # 定义一个测试方法，用于测试 Timedelta 类中的 round、floor 和 ceil 方法的正确性
    def test_round_sanity(self, val, method):
        # 别名 cls 用于引用 Timedelta 类
        cls = Timedelta
        # 定义异常类 OutOfBoundsTimedelta 的别名 err_cls
        err_cls = OutOfBoundsTimedelta

        # 将 val 转换为 np.int64 类型
        val = np.int64(val)
        # 使用 Timedelta 类创建一个时间差实例 td
        td = cls(val)

        # 定义内部函数 checker，用于检查不同时间单位下的 round、floor 和 ceil 方法的行为
        def checker(ts, nanos, unit):
            # 首先检查是否应该引发异常的情况
            if nanos == 1:
                # 如果 nanos 等于 1，则跳过检查
                pass
            else:
                # 否则，计算时间戳 ts 与 nanos 的商和余数
                div, mod = divmod(ts._value, nanos)
                # 计算差值
                diff = int(nanos - mod)
                # 计算下界 lb 和上界 ub
                lb = ts._value - mod
                # 断言下界 lb 小于等于时间戳 ts 的值，以确保没有 Python 整数溢出
                assert lb <= ts._value
                # 计算上界 ub
                ub = ts._value + diff
                # 断言上界 ub 大于时间戳 ts 的值，以确保没有 Python 整数溢出
                assert ub > ts._value

                # 定义消息 "without overflow"
                msg = "without overflow"
                if mod == 0:
                    # 如果 mod 等于 0，则不应该引发异常，跳过检查
                    pass
                elif method is cls.ceil:
                    # 如果 method 是 ceil 方法，且 ub 大于 Timedelta 类的最大值，则使用 pytest 的 raises 断言引发 err_cls 异常
                    if ub > cls.max._value:
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif method is cls.floor:
                    # 如果 method 是 floor 方法，且 lb 小于 Timedelta 类的最小值，则使用 pytest 的 raises 断言引发 err_cls 异常
                    if lb < cls.min._value:
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif mod >= diff:
                    # 如果 mod 大于等于 diff，则使用 pytest 的 raises 断言引发 err_cls 异常
                    if ub > cls.max._value:
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif lb < cls.min._value:
                    # 如果 lb 小于 Timedelta 类的最小值，则使用 pytest 的 raises 断言引发 err_cls 异常
                    with pytest.raises(err_cls, match=msg):
                        method(ts, unit)
                    return

            # 执行 method 方法，获取结果 res
            res = method(ts, unit)

            # 计算时间差 td
            td = res - ts
            # 计算绝对差值 diff
            diff = abs(td._value)
            # 断言 diff 小于 nanos
            assert diff < nanos
            # 断言 res 的值可以被 nanos 整除
            assert res._value % nanos == 0

            # 根据不同的 method 方法进行进一步的断言
            if method is cls.round:
                # 如果 method 是 round 方法，则断言 diff 小于等于 nanos 的一半
                assert diff <= nanos / 2
            elif method is cls.floor:
                # 如果 method 是 floor 方法，则断言 res 小于等于 ts
                assert res <= ts
            elif method is cls.ceil:
                # 如果 method 是 ceil 方法，则断言 res 大于等于 ts
                assert res >= ts

        # 测试不同的时间单位下的 checker 函数
        nanos = 1
        checker(td, nanos, "ns")

        nanos = 1000
        checker(td, nanos, "us")

        nanos = 1_000_000
        checker(td, nanos, "ms")

        nanos = 1_000_000_000
        checker(td, nanos, "s")

        nanos = 60 * 1_000_000_000
        checker(td, nanos, "min")

        nanos = 60 * 60 * 1_000_000_000
        checker(td, nanos, "h")

        nanos = 24 * 60 * 60 * 1_000_000_000
        checker(td, nanos, "D")
    # 定义一个测试方法，测试时间差对象的舍入、向下取整和向上取整功能
    def test_round_non_nano(self, unit):
        # 创建一个时间差对象，表示"1 days 02:34:57"，并将其转换为指定单位（unit）
        td = Timedelta("1 days 02:34:57").as_unit(unit)

        # 对时间差对象进行分钟级别的舍入操作，预期结果是将秒舍入到最接近的整分钟
        res = td.round("min")
        # 断言舍入后的结果为"1 days 02:35:00"
        assert res == Timedelta("1 days 02:35:00")
        # 断言舍入后的结果的分辨率（_resolution，缩写为_creso）与原时间差对象相同
        assert res._creso == td._creso

        # 对时间差对象进行分钟级别的向下取整操作，预期结果是将秒数向下舍去，保留整分钟
        res = td.floor("min")
        # 断言向下取整后的结果为"1 days 02:34:00"
        assert res == Timedelta("1 days 02:34:00")
        # 断言向下取整后的结果的分辨率与原时间差对象相同
        assert res._creso == td._creso

        # 对时间差对象进行分钟级别的向上取整操作，预期结果是将秒数向上舍入，保留整分钟
        res = td.ceil("min")
        # 断言向上取整后的结果为"1 days 02:35:00"
        assert res == Timedelta("1 days 02:35:00")
        # 断言向上取整后的结果的分辨率与原时间差对象相同
        assert res._creso == td._creso
```