# `D:\src\scipysrc\pandas\pandas\tests\scalar\timestamp\methods\test_as_unit.py`

```
import pytest  # 导入 pytest 测试框架

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit  # 从 pandas 库导入 NpyDatetimeUnit 类型
from pandas.errors import OutOfBoundsDatetime  # 从 pandas.errors 导入 OutOfBoundsDatetime 异常类

from pandas import Timestamp  # 从 pandas 库导入 Timestamp 类


class TestTimestampAsUnit:
    def test_as_unit(self):
        ts = Timestamp("1970-01-01").as_unit("ns")  # 创建 Timestamp 对象，并转换为纳秒单位
        assert ts.unit == "ns"  # 断言 Timestamp 对象的单位为纳秒

        assert ts.as_unit("ns") is ts  # 断言再次转换为纳秒后仍为同一对象

        res = ts.as_unit("us")  # 转换为微秒单位
        assert res._value == ts._value // 1000  # 断言微秒值与纳秒值的转换关系
        assert res._creso == NpyDatetimeUnit.NPY_FR_us.value  # 断言单位为微秒

        rt = res.as_unit("ns")  # 再次转换为纳秒单位
        assert rt._value == ts._value  # 断言转换后的纳秒值与原始值相同
        assert rt._creso == ts._creso  # 断言转换后的单位保持不变

        res = ts.as_unit("ms")  # 转换为毫秒单位
        assert res._value == ts._value // 1_000_000  # 断言毫秒值与纳秒值的转换关系
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value  # 断言单位为毫秒

        rt = res.as_unit("ns")  # 再次转换为纳秒单位
        assert rt._value == ts._value  # 断言转换后的纳秒值与原始值相同
        assert rt._creso == ts._creso  # 断言转换后的单位保持不变

        res = ts.as_unit("s")  # 转换为秒单位
        assert res._value == ts._value // 1_000_000_000  # 断言秒值与纳秒值的转换关系
        assert res._creso == NpyDatetimeUnit.NPY_FR_s.value  # 断言单位为秒

        rt = res.as_unit("ns")  # 再次转换为纳秒单位
        assert rt._value == ts._value  # 断言转换后的纳秒值与原始值相同
        assert rt._creso == ts._creso  # 断言转换后的单位保持不变

    def test_as_unit_overflows(self):
        # 创建一个微秒级别的 Timestamp 对象，使其在转换为纳秒时超出界限
        us = 9223372800000000
        ts = Timestamp._from_value_and_reso(us, NpyDatetimeUnit.NPY_FR_us.value, None)

        msg = "Cannot cast 2262-04-12 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsDatetime, match=msg):  # 断言转换为纳秒单位时引发 OutOfBoundsDatetime 异常
            ts.as_unit("ns")

        res = ts.as_unit("ms")  # 转换为毫秒单位
        assert res._value == us // 1000  # 断言毫秒值与微秒值的转换关系
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value  # 断言单位为毫秒

    def test_as_unit_rounding(self):
        ts = Timestamp(1_500_000)  # 创建一个 Timestamp 对象，代表 1500 微秒
        res = ts.as_unit("ms")  # 转换为毫秒单位

        expected = Timestamp(1_000_000)  # 预期结果是 1 毫秒
        assert res == expected  # 断言转换结果与预期相同

        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value  # 断言单位为毫秒
        assert res._value == 1  # 断言值为 1

        with pytest.raises(ValueError, match="Cannot losslessly convert units"):  # 断言在不可损失转换单位时引发 ValueError
            ts.as_unit("ms", round_ok=False)

    def test_as_unit_non_nano(self):
        # 创建一个 Timestamp 对象，先转换为毫秒单位，然后再转换为秒单位
        ts = Timestamp("1970-01-02").as_unit("ms")
        assert ts.year == 1970  # 断言年份为 1970
        assert ts.month == 1  # 断言月份为 1
        assert ts.day == 2  # 断言日期为 2
        assert ts.hour == ts.minute == ts.second == ts.microsecond == ts.nanosecond == 0  # 断言时间为 0

        res = ts.as_unit("s")  # 再次转换为秒单位
        assert res._value == 24 * 3600  # 断言秒值为一天的秒数
        assert res.year == 1970  # 断言年份为 1970
        assert res.month == 1  # 断言月份为 1
        assert res.day == 2  # 断言日期为 2
        assert (
            res.hour
            == res.minute
            == res.second
            == res.microsecond
            == res.nanosecond
            == 0
        )  # 断言时间为 0
```