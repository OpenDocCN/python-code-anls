# `D:\src\scipysrc\pandas\pandas\tests\scalar\timedelta\methods\test_as_unit.py`

```
import pytest  # 导入 pytest 库

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit  # 从 pandas 库中导入 NpyDatetimeUnit 类型
from pandas.errors import OutOfBoundsTimedelta  # 从 pandas 库中导入 OutOfBoundsTimedelta 异常类

from pandas import Timedelta  # 从 pandas 库中导入 Timedelta 类


class TestAsUnit:
    def test_as_unit(self):
        td = Timedelta(days=1)  # 创建一个 Timedelta 对象，表示一天的时间间隔

        assert td.as_unit("ns") is td  # 断言将时间间隔单位转换为纳秒后仍然返回原对象

        res = td.as_unit("us")  # 将时间间隔单位转换为微秒
        assert res._value == td._value // 1000  # 断言转换后的值等于原值除以1000
        assert res._creso == NpyDatetimeUnit.NPY_FR_us.value  # 断言转换后的分辨率为微秒

        rt = res.as_unit("ns")  # 将时间间隔单位再次转换回纳秒
        assert rt._value == td._value  # 断言转换后的值与原值相等
        assert rt._creso == td._creso  # 断言转换后的分辨率与原对象相同

        res = td.as_unit("ms")  # 将时间间隔单位转换为毫秒
        assert res._value == td._value // 1_000_000  # 断言转换后的值等于原值除以1000000
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value  # 断言转换后的分辨率为毫秒

        rt = res.as_unit("ns")  # 将时间间隔单位再次转换回纳秒
        assert rt._value == td._value  # 断言转换后的值与原值相等
        assert rt._creso == td._creso  # 断言转换后的分辨率与原对象相同

        res = td.as_unit("s")  # 将时间间隔单位转换为秒
        assert res._value == td._value // 1_000_000_000  # 断言转换后的值等于原值除以1000000000
        assert res._creso == NpyDatetimeUnit.NPY_FR_s.value  # 断言转换后的分辨率为秒

        rt = res.as_unit("ns")  # 将时间间隔单位再次转换回纳秒
        assert rt._value == td._value  # 断言转换后的值与原值相等
        assert rt._creso == td._creso  # 断言转换后的分辨率与原对象相同

    def test_as_unit_overflows(self):
        # 定义一个超出纳秒单位范围的微秒值
        us = 9223372800000000
        # 创建一个 Timedelta 对象，表示该微秒值和分辨率为微秒
        td = Timedelta._from_value_and_reso(us, NpyDatetimeUnit.NPY_FR_us.value)

        msg = "Cannot cast 106752 days 00:00:00 to unit='ns' without overflow"
        # 使用 pytest 断言捕获 OutOfBoundsTimedelta 异常，并验证错误消息匹配特定模式
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            td.as_unit("ns")

        res = td.as_unit("ms")  # 将时间间隔单位转换为毫秒
        assert res._value == us // 1000  # 断言转换后的值等于原值除以1000
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value  # 断言转换后的分辨率为毫秒

    def test_as_unit_rounding(self):
        td = Timedelta(microseconds=1500)  # 创建一个微秒为1500的 Timedelta 对象
        res = td.as_unit("ms")  # 将时间间隔单位转换为毫秒

        expected = Timedelta(milliseconds=1)  # 创建一个预期的 Timedelta 对象，表示1毫秒
        assert res == expected  # 断言转换后的结果与预期结果相等

        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value  # 断言转换后的分辨率为毫秒
        assert res._value == 1  # 断言转换后的值为1

        # 使用 pytest 断言捕获 ValueError 异常，并验证错误消息匹配特定模式
        with pytest.raises(ValueError, match="Cannot losslessly convert units"):
            td.as_unit("ms", round_ok=False)

    def test_as_unit_non_nano(self):
        # 将一天的时间间隔转换为毫秒
        td = Timedelta(days=1).as_unit("ms")
        assert td.days == 1  # 断言转换后的天数为1
        assert td._value == 86_400_000  # 断言转换后的值为86400000
        assert td.components.days == 1  # 断言转换后的天数为1
        assert td._d == 1  # 断言转换后的天数为1
        assert td.total_seconds() == 86400  # 断言转换后的总秒数为86400

        res = td.as_unit("us")  # 将时间间隔单位再次转换为微秒
        assert res._value == 86_400_000_000  # 断言转换后的值为86400000000
        assert res.components.days == 1  # 断言转换后的天数为1
        assert res.components.hours == 0  # 断言转换后的小时数为0
        assert res._d == 1  # 断言转换后的天数为1
        assert res._h == 0  # 断言转换后的小时数为0
        assert res.total_seconds() == 86400  # 断言转换后的总秒数为86400
```