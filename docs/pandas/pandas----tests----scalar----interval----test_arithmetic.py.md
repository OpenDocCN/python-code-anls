# `D:\src\scipysrc\pandas\pandas\tests\scalar\interval\test_arithmetic.py`

```
from datetime import timedelta  # 导入timedelta类，用于处理时间差

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试

from pandas import (  # 从pandas库导入以下模块：
    Interval,  # Interval表示时间间隔
    Timedelta,  # Timedelta表示时间差
    Timestamp,  # Timestamp表示时间戳
)
import pandas._testing as tm  # 导入pandas内部测试模块

class TestIntervalArithmetic:
    def test_interval_add(self, closed):
        interval = Interval(0, 1, closed=closed)  # 创建一个闭区间[0, 1]
        expected = Interval(1, 2, closed=closed)  # 预期结果是闭区间[1, 2]

        result = interval + 1  # 对区间进行加法运算
        assert result == expected  # 断言结果符合预期

        result = 1 + interval  # 另一种加法运算方式
        assert result == expected  # 断言结果符合预期

        result = interval
        result += 1  # 原地加法运算
        assert result == expected  # 断言结果符合预期

        msg = r"unsupported operand type\(s\) for \+"  # 出现不支持的操作类型错误信息
        with pytest.raises(TypeError, match=msg):  # 使用pytest断言捕获预期的TypeError异常
            interval + interval  # 试图对区间与区间进行加法操作

        with pytest.raises(TypeError, match=msg):  # 同样捕获预期的TypeError异常
            interval + "foo"  # 试图对区间与字符串进行加法操作

    def test_interval_sub(self, closed):
        interval = Interval(0, 1, closed=closed)  # 创建一个闭区间[0, 1]
        expected = Interval(-1, 0, closed=closed)  # 预期结果是闭区间[-1, 0]

        result = interval - 1  # 区间减法运算
        assert result == expected  # 断言结果符合预期

        result = interval
        result -= 1  # 原地减法运算
        assert result == expected  # 断言结果符合预期

        msg = r"unsupported operand type\(s\) for -"  # 出现不支持的操作类型错误信息
        with pytest.raises(TypeError, match=msg):  # 使用pytest断言捕获预期的TypeError异常
            interval - interval  # 试图对区间与区间进行减法操作

        with pytest.raises(TypeError, match=msg):  # 同样捕获预期的TypeError异常
            interval - "foo"  # 试图对区间与字符串进行减法操作

    def test_interval_mult(self, closed):
        interval = Interval(0, 1, closed=closed)  # 创建一个闭区间[0, 1]
        expected = Interval(0, 2, closed=closed)  # 预期结果是闭区间[0, 2]

        result = interval * 2  # 区间乘法运算
        assert result == expected  # 断言结果符合预期

        result = 2 * interval  # 另一种乘法运算方式
        assert result == expected  # 断言结果符合预期

        result = interval
        result *= 2  # 原地乘法运算
        assert result == expected  # 断言结果符合预期

        msg = r"unsupported operand type\(s\) for \*"  # 出现不支持的操作类型错误信息
        with pytest.raises(TypeError, match=msg):  # 使用pytest断言捕获预期的TypeError异常
            interval * interval  # 试图对区间与区间进行乘法操作

        msg = r"can\'t multiply sequence by non-int"  # 出现无法将序列乘以非整数错误信息
        with pytest.raises(TypeError, match=msg):  # 使用pytest断言捕获预期的TypeError异常
            interval * "foo"  # 试图对区间与字符串进行乘法操作

    def test_interval_div(self, closed):
        interval = Interval(0, 1, closed=closed)  # 创建一个闭区间[0, 1]
        expected = Interval(0, 0.5, closed=closed)  # 预期结果是闭区间[0, 0.5]

        result = interval / 2.0  # 区间除法运算
        assert result == expected  # 断言结果符合预期

        result = interval
        result /= 2.0  # 原地除法运算
        assert result == expected  # 断言结果符合预期

        msg = r"unsupported operand type\(s\) for /"  # 出现不支持的操作类型错误信息
        with pytest.raises(TypeError, match=msg):  # 使用pytest断言捕获预期的TypeError异常
            interval / interval  # 试图对区间与区间进行除法操作

        with pytest.raises(TypeError, match=msg):  # 同样捕获预期的TypeError异常
            interval / "foo"  # 试图对区间与字符串进行除法操作

    def test_interval_floordiv(self, closed):
        interval = Interval(1, 2, closed=closed)  # 创建一个闭区间[1, 2]
        expected = Interval(0, 1, closed=closed)  # 预期结果是闭区间[0, 1]

        result = interval // 2  # 区间整除运算
        assert result == expected  # 断言结果符合预期

        result = interval
        result //= 2  # 原地整除运算
        assert result == expected  # 断言结果符合预期

        msg = r"unsupported operand type\(s\) for //"  # 出现不支持的操作类型错误信息
        with pytest.raises(TypeError, match=msg):  # 使用pytest断言捕获预期的TypeError异常
            interval // interval  # 试图对区间与区间进行整除操作

        with pytest.raises(TypeError, match=msg):  # 同样捕获预期的TypeError异常
            interval // "foo"  # 试图对区间与字符串进行整除操作

    @pytest.mark.parametrize("method", ["__add__", "__sub__"])
    # 使用 pytest 的参数化装饰器，为测试方法 test_time_interval_add_subtract_timedelta 添加多组参数
    @pytest.mark.parametrize(
        "interval",
        [
            Interval(
                Timestamp("2017-01-01 00:00:00"), Timestamp("2018-01-01 00:00:00")
            ),  # 创建一个时间区间对象，起始和结束时间分别为指定的时间戳
            Interval(Timedelta(days=7), Timedelta(days=14)),  # 创建一个时间区间对象，起始和结束时间分别为指定的时间增量
        ],
    )
    @pytest.mark.parametrize(
        "delta", [Timedelta(days=7), timedelta(7), np.timedelta64(7, "D")]
    )
    # 定义测试方法，用于测试时间区间对象与时间增量之间的加减法操作
    def test_time_interval_add_subtract_timedelta(self, interval, delta, method):
        # 发现的问题的 GitHub 链接
        result = getattr(interval, method)(delta)  # 执行时间区间对象的指定方法，对时间增量进行操作，得到结果
        left = getattr(interval.left, method)(delta)  # 对时间区间起始时间进行相同的操作，得到左端点的新值
        right = getattr(interval.right, method)(delta)  # 对时间区间结束时间进行相同的操作，得到右端点的新值
        expected = Interval(left, right)  # 创建一个新的时间区间对象，其起始和结束时间为左右端点的新值
    
        assert result == expected  # 断言实际结果与期望结果相等
    
    # 使用 pytest 的参数化装饰器，为测试方法 test_numeric_interval_add_timedelta_raises 添加多组参数
    @pytest.mark.parametrize("interval", [Interval(1, 2), Interval(1.0, 2.0)])
    @pytest.mark.parametrize(
        "delta", [Timedelta(days=7), timedelta(7), np.timedelta64(7, "D")]
    )
    # 定义测试方法，测试在给定条件下，将时间区间对象与非数值类型的增量相加会引发错误
    def test_numeric_interval_add_timedelta_raises(self, interval, delta):
        # 发现的问题的 GitHub 链接
        msg = "|".join(
            [
                "unsupported operand",
                "cannot use operands",
                "Only numeric, Timestamp and Timedelta endpoints are allowed",
            ]
        )
        with pytest.raises((TypeError, ValueError), match=msg):
            interval + delta  # 断言在加法操作中会引发 TypeError 或 ValueError 异常，异常信息符合预期
    
        with pytest.raises((TypeError, ValueError), match=msg):
            delta + interval  # 断言在反向加法操作中会引发 TypeError 或 ValueError 异常，异常信息符合预期
    
    # 使用 pytest 的参数化装饰器，为测试方法 test_timedelta_add_timestamp_interval 添加多组参数
    @pytest.mark.parametrize("klass", [timedelta, np.timedelta64, Timedelta])
    # 定义测试方法，测试时间增量与时间戳组成的时间区间对象的加法操作
    def test_timedelta_add_timestamp_interval(self, klass):
        delta = klass(0)  # 创建一个指定类型的时间增量对象
        expected = Interval(Timestamp("2020-01-01"), Timestamp("2020-02-01"))  # 创建一个时间区间对象，起始和结束时间分别为指定的时间戳
    
        result = delta + expected  # 执行时间增量与时间区间对象的加法操作
        assert result == expected  # 断言加法操作的结果与期望结果相等
    
        result = expected + delta  # 执行时间区间对象与时间增量的反向加法操作
        assert result == expected  # 断言反向加法操作的结果与期望结果相等
class TestIntervalComparisons:
    def test_interval_equal(self):
        # 检查两个区间是否相等，右端点包含
        assert Interval(0, 1) == Interval(0, 1, closed="right")
        # 检查两个区间是否不相等，左端点包含
        assert Interval(0, 1) != Interval(0, 1, closed="left")
        # 检查区间与整数是否不相等
        assert Interval(0, 1) != 0

    def test_interval_comparison(self):
        # 准备异常消息字符串
        msg = (
            "'<' not supported between instances of "
            "'pandas._libs.interval.Interval' and 'int'"
        )
        # 确保当区间与整数比较时会引发类型错误异常，异常消息需匹配指定的消息
        with pytest.raises(TypeError, match=msg):
            Interval(0, 1) < 2

        # 检查区间之间的大小比较
        assert Interval(0, 1) < Interval(1, 2)
        assert Interval(0, 1) < Interval(0, 2)
        assert Interval(0, 1) < Interval(0.5, 1.5)
        assert Interval(0, 1) <= Interval(0, 1)
        assert Interval(0, 1) > Interval(-1, 2)
        assert Interval(0, 1) >= Interval(0, 1)

    def test_equality_comparison_broadcasts_over_array(self):
        # 在数组中广播比较区间相等性的情况
        # 参考：https://github.com/pandas-dev/pandas/issues/35931
        interval = Interval(0, 1)
        arr = np.array([interval, interval])
        result = interval == arr
        expected = np.array([True, True])
        # 断言 numpy 数组相等
        tm.assert_numpy_array_equal(result, expected)
```