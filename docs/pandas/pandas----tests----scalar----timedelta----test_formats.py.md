# `D:\src\scipysrc\pandas\pandas\tests\scalar\timedelta\test_formats.py`

```
import pytest  # 导入 pytest 库

from pandas import Timedelta  # 从 pandas 库中导入 Timedelta 类


@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，用来定义多组参数化测试数据
    "td, expected_repr",  # 定义参数名称
    [  # 参数化测试数据列表
        (Timedelta(10, unit="D"), "Timedelta('10 days 00:00:00')"),  # 测试 Timedelta 对象表示天数
        (Timedelta(10, unit="s"), "Timedelta('0 days 00:00:10')"),  # 测试 Timedelta 对象表示秒数
        (Timedelta(10, unit="ms"), "Timedelta('0 days 00:00:00.010000')"),  # 测试 Timedelta 对象表示毫秒数
        (Timedelta(-10, unit="ms"), "Timedelta('-1 days +23:59:59.990000')"),  # 测试负 Timedelta 对象表示毫秒数
    ],
)
def test_repr(td, expected_repr):  # 定义测试函数 test_repr，用来测试 Timedelta 对象的字符串表示
    assert repr(td) == expected_repr  # 断言 Timedelta 对象的 repr 方法输出符合预期的字符串表示


@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，定义第二组参数化测试数据
    "td, expected_iso",  # 定义参数名称
    [  # 参数化测试数据列表
        (
            Timedelta(
                days=6,
                minutes=50,
                seconds=3,
                milliseconds=10,
                microseconds=10,
                nanoseconds=12,
            ),
            "P6DT0H50M3.010010012S",  # 测试 Timedelta 对象的 ISO 格式表示
        ),
        (Timedelta(days=4, hours=12, minutes=30, seconds=5), "P4DT12H30M5S"),  # 测试 Timedelta 对象的 ISO 格式表示
        (Timedelta(nanoseconds=123), "P0DT0H0M0.000000123S"),  # 测试 Timedelta 对象的 ISO 格式表示
        # trim nano
        (Timedelta(microseconds=10), "P0DT0H0M0.00001S"),  # 测试 Timedelta 对象的 ISO 格式表示
        # trim micro
        (Timedelta(milliseconds=1), "P0DT0H0M0.001S"),  # 测试 Timedelta 对象的 ISO 格式表示
        # don't strip every 0
        (Timedelta(minutes=1), "P0DT0H1M0S"),  # 测试 Timedelta 对象的 ISO 格式表示
    ],
)
def test_isoformat(td, expected_iso):  # 定义测试函数 test_isoformat，用来测试 Timedelta 对象的 ISO 格式化表示
    assert td.isoformat() == expected_iso  # 断言 Timedelta 对象的 isoformat 方法输出符合预期的 ISO 格式字符串


class TestReprBase:  # 定义测试类 TestReprBase
    def test_none(self):  # 定义测试方法 test_none
        delta_1d = Timedelta(1, unit="D")  # 创建 Timedelta 对象表示一天
        delta_0d = Timedelta(0, unit="D")  # 创建 Timedelta 对象表示零天
        delta_1s = Timedelta(1, unit="s")  # 创建 Timedelta 对象表示一秒
        delta_500ms = Timedelta(500, unit="ms")  # 创建 Timedelta 对象表示500毫秒

        drepr = lambda x: x._repr_base()  # 定义 lambda 函数 drepr，用于获取 Timedelta 对象的基本表示
        assert drepr(delta_1d) == "1 days"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(-delta_1d) == "-1 days"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(delta_0d) == "0 days"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(delta_1s) == "0 days 00:00:01"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(delta_500ms) == "0 days 00:00:00.500000"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(delta_1d + delta_1s) == "1 days 00:00:01"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(-delta_1d + delta_1s) == "-1 days +00:00:01"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(delta_1d + delta_500ms) == "1 days 00:00:00.500000"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(-delta_1d + delta_500ms) == "-1 days +00:00:00.500000"  # 断言 Timedelta 对象的基本表示符合预期

    def test_sub_day(self):  # 定义测试方法 test_sub_day
        delta_1d = Timedelta(1, unit="D")  # 创建 Timedelta 对象表示一天
        delta_0d = Timedelta(0, unit="D")  # 创建 Timedelta 对象表示零天
        delta_1s = Timedelta(1, unit="s")  # 创建 Timedelta 对象表示一秒
        delta_500ms = Timedelta(500, unit="ms")  # 创建 Timedelta 对象表示500毫秒

        drepr = lambda x: x._repr_base(format="sub_day")  # 定义 lambda 函数 drepr，用于获取 Timedelta 对象的基本表示（按天减少）
        assert drepr(delta_1d) == "1 days"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(-delta_1d) == "-1 days"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(delta_0d) == "00:00:00"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(delta_1s) == "00:00:01"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(delta_500ms) == "00:00:00.500000"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(delta_1d + delta_1s) == "1 days 00:00:01"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(-delta_1d + delta_1s) == "-1 days +00:00:01"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(delta_1d + delta_500ms) == "1 days 00:00:00.500000"  # 断言 Timedelta 对象的基本表示符合预期
        assert drepr(-delta_1d + delta_500ms) == "-1 days +00:00:00.500000"  # 断言 Timedelta 对象的基本表示符合预期
    # 定义测试方法，验证 Timedelta 类的长格式字符串表示功能
    def test_long(self):
        # 创建 Timedelta 对象，表示1天
        delta_1d = Timedelta(1, unit="D")
        # 创建 Timedelta 对象，表示0天
        delta_0d = Timedelta(0, unit="D")
        # 创建 Timedelta 对象，表示1秒
        delta_1s = Timedelta(1, unit="s")
        # 创建 Timedelta 对象，表示500毫秒
        delta_500ms = Timedelta(500, unit="ms")

        # 定义 lambda 函数 drepr，用于获取 Timedelta 对象的长格式字符串表示
        drepr = lambda x: x._repr_base(format="long")
        
        # 断言：验证 Timedelta 对象的长格式字符串表示是否符合预期
        assert drepr(delta_1d) == "1 days 00:00:00"
        assert drepr(-delta_1d) == "-1 days +00:00:00"
        assert drepr(delta_0d) == "0 days 00:00:00"
        assert drepr(delta_1s) == "0 days 00:00:01"
        assert drepr(delta_500ms) == "0 days 00:00:00.500000"
        assert drepr(delta_1d + delta_1s) == "1 days 00:00:01"
        assert drepr(-delta_1d + delta_1s) == "-1 days +00:00:01"
        assert drepr(delta_1d + delta_500ms) == "1 days 00:00:00.500000"
        assert drepr(-delta_1d + delta_500ms) == "-1 days +00:00:00.500000"

    # 定义测试方法，验证 Timedelta 类的全部精度格式字符串表示功能
    def test_all(self):
        # 创建 Timedelta 对象，表示1天
        delta_1d = Timedelta(1, unit="D")
        # 创建 Timedelta 对象，表示0天
        delta_0d = Timedelta(0, unit="D")
        # 创建 Timedelta 对象，表示1纳秒
        delta_1ns = Timedelta(1, unit="ns")

        # 定义 lambda 函数 drepr，用于获取 Timedelta 对象的全部精度格式字符串表示
        drepr = lambda x: x._repr_base(format="all")
        
        # 断言：验证 Timedelta 对象的全部精度格式字符串表示是否符合预期
        assert drepr(delta_1d) == "1 days 00:00:00.000000000"
        assert drepr(-delta_1d) == "-1 days +00:00:00.000000000"
        assert drepr(delta_0d) == "0 days 00:00:00.000000000"
        assert drepr(delta_1ns) == "0 days 00:00:00.000000001"
        assert drepr(-delta_1d + delta_1ns) == "-1 days +00:00:00.000000001"
```