# `D:\src\scipysrc\pandas\pandas\tests\scalar\interval\test_constructors.py`

```
import pytest  # 导入 pytest 模块

from pandas import (  # 从 pandas 库中导入以下模块：
    Interval,  # 区间 Interval
    Period,    # 时间段 Period
    Timestamp,  # 时间戳 Timestamp
)


class TestIntervalConstructors:  # 定义测试类 TestIntervalConstructors
    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，设置测试参数
        "left, right",  # 参数化的参数名称
        [  # 参数化的具体参数组合列表
            ("a", "z"),  # 字符串参数组合
            (("a", "b"), ("c", "d")),  # 元组参数组合
            (list("AB"), list("ab")),  # 列表参数组合
            (Interval(0, 1), Interval(1, 2)),  # Interval 对象参数组合
            (Period("2018Q1", freq="Q"), Period("2018Q1", freq="Q")),  # Period 对象参数组合
        ],
    )
    def test_construct_errors(self, left, right):  # 定义测试方法 test_construct_errors
        # GH#23013
        msg = "Only numeric, Timestamp and Timedelta endpoints are allowed"  # 设置错误消息
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 的断言检查是否引发 ValueError 异常并匹配消息
            Interval(left, right)  # 创建 Interval 对象

    def test_constructor_errors(self):  # 定义测试方法 test_constructor_errors
        msg = "invalid option for 'closed': foo"  # 设置错误消息
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 的断言检查是否引发 ValueError 异常并匹配消息
            Interval(0, 1, closed="foo")  # 创建 Interval 对象

        msg = "left side of interval must be <= right side"  # 设置错误消息
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 的断言检查是否引发 ValueError 异常并匹配消息
            Interval(1, 0)  # 创建 Interval 对象

    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，设置测试参数
        "tz_left, tz_right", [(None, "UTC"), ("UTC", None), ("UTC", "US/Eastern")]
    )
    def test_constructor_errors_tz(self, tz_left, tz_right):  # 定义测试方法 test_constructor_errors_tz
        # GH#18538
        left = Timestamp("2017-01-01", tz=tz_left)  # 创建左侧 Timestamp 对象
        right = Timestamp("2017-01-02", tz=tz_right)  # 创建右侧 Timestamp 对象

        if tz_left is None or tz_right is None:  # 如果左侧或右侧时区为 None
            error = TypeError  # 错误类型为 TypeError
            msg = "Cannot compare tz-naive and tz-aware timestamps"  # 设置错误消息
        else:  # 否则
            error = ValueError  # 错误类型为 ValueError
            msg = "left and right must have the same time zone"  # 设置错误消息
        with pytest.raises(error, match=msg):  # 使用 pytest 的断言检查是否引发相应类型的异常并匹配消息
            Interval(left, right)  # 创建 Interval 对象
```