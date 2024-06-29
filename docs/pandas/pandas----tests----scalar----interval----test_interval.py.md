# `D:\src\scipysrc\pandas\pandas\tests\scalar\interval\test_interval.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值计算
import pytest  # 导入 Pytest 库，用于单元测试

from pandas import (  # 从 Pandas 库中导入以下几个类
    Interval,  # 时间间隔类
    Timedelta,  # 时间增量类
    Timestamp,  # 时间戳类
)


@pytest.fixture
def interval():  # 定义一个 Pytest 的 fixture，返回一个 Interval 对象
    return Interval(0, 1)  # 创建并返回一个 Interval 对象，表示闭区间 [0, 1]


class TestInterval:  # 定义一个测试类 TestInterval
    def test_properties(self, interval):  # 测试 Interval 对象的属性
        assert interval.closed == "right"  # 检查闭区间的属性是否为 "right"
        assert interval.left == 0  # 检查区间左端点是否为 0
        assert interval.right == 1  # 检查区间右端点是否为 1
        assert interval.mid == 0.5  # 检查区间中点是否为 0.5

    def test_hash(self, interval):  # 测试 Interval 对象的哈希值
        # 应该不会引发异常
        hash(interval)

    @pytest.mark.parametrize(  # 使用 Pytest 的参数化装饰器定义多组参数
        "left, right, expected",  # 参数名
        [  # 参数组列表
            (0, 5, 5),  # 第一组参数
            (-2, 5.5, 7.5),  # 第二组参数
            (10, 10, 0),  # 第三组参数
            (10, np.inf, np.inf),  # 第四组参数
            (-np.inf, -5, np.inf),  # 第五组参数
            (-np.inf, np.inf, np.inf),  # 第六组参数
            (Timedelta("0 days"), Timedelta("5 days"), Timedelta("5 days")),  # 第七组参数
            (Timedelta("10 days"), Timedelta("10 days"), Timedelta("0 days")),  # 第八组参数
            (Timedelta("1h10min"), Timedelta("5h5min"), Timedelta("3h55min")),  # 第九组参数
            (Timedelta("5s"), Timedelta("1h"), Timedelta("59min55s")),  # 第十组参数
        ],
    )
    def test_length(self, left, right, expected):  # 测试 Interval 长度计算方法
        # GH 18789
        iv = Interval(left, right)  # 创建一个 Interval 对象
        result = iv.length  # 计算 Interval 的长度
        assert result == expected  # 断言计算结果是否等于预期值

    @pytest.mark.parametrize(  # 使用 Pytest 的参数化装饰器定义多组参数
        "left, right, expected",  # 参数名
        [  # 参数组列表
            ("2017-01-01", "2017-01-06", "5 days"),  # 第一组参数
            ("2017-01-01", "2017-01-01 12:00:00", "12 hours"),  # 第二组参数
            ("2017-01-01 12:00", "2017-01-01 12:00:00", "0 days"),  # 第三组参数
            ("2017-01-01 12:01", "2017-01-05 17:31:00", "4 days 5 hours 30 min"),  # 第四组参数
        ],
    )
    @pytest.mark.parametrize("tz", (None, "UTC", "CET", "US/Eastern"))  # 参数化时区参数
    def test_length_timestamp(self, tz, left, right, expected):  # 测试 Timestamp 间隔长度计算
        # GH 18789
        iv = Interval(Timestamp(left, tz=tz), Timestamp(right, tz=tz))  # 创建一个 Timestamp 区间对象
        result = iv.length  # 计算区间长度
        expected = Timedelta(expected)  # 转换预期结果为 Timedelta 对象
        assert result == expected  # 断言计算结果是否等于预期值

    @pytest.mark.parametrize(  # 使用 Pytest 的参数化装饰器定义多组参数
        "left, right",  # 参数名
        [  # 参数组列表
            (0, 1),  # 第一组参数
            (Timedelta("0 days"), Timedelta("1 day")),  # 第二组参数
            (Timestamp("2018-01-01"), Timestamp("2018-01-02")),  # 第三组参数
            (  # 第四组参数，包含时区信息的 Timestamp
                Timestamp("2018-01-01", tz="US/Eastern"),
                Timestamp("2018-01-02", tz="US/Eastern"),
            ),
        ],
    )
    def test_is_empty(self, left, right, closed):  # 测试区间是否为空的方法
        # GH27219
        # 非空区间应始终返回 False
        iv = Interval(left, right, closed)  # 创建一个 Interval 对象
        assert iv.is_empty is False  # 断言非空区间应返回 False

        # 相同端点为空，除非 closed='both'（包含一个点）
        iv = Interval(left, left, closed)  # 创建一个左右端点相同的 Interval 对象
        result = iv.is_empty  # 计算区间是否为空
        expected = closed != "both"  # 计算预期结果
        assert result is expected  # 断言计算结果是否等于预期值
```